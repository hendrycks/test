# coding=utf-8
# Copyright 2020 The Uncertainty Metrics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""General metric defining the parameterized space of calibration metrics.
"""

import itertools
import numpy as np


def one_hot_encode(labels, num_classes=None):
  """One hot encoder for turning a vector of labels into a OHE matrix."""
  if num_classes is None:
    num_classes = len(np.unique(labels))
  return np.eye(num_classes)[labels]


def mean(inputs):
  """Be able to take the mean of an empty array without hitting NANs."""
  # pylint disable necessary for numpy and pandas
  if len(inputs) == 0:  # pylint: disable=g-explicit-length-test
    return 0
  else:
    return np.mean(inputs)


def get_adaptive_bins(predictions, num_bins):
  """Returns upper edges for binning an equal number of datapoints per bin."""
  if np.size(predictions) == 0:
    return np.linspace(0, 1, num_bins+1)[:-1]

  edge_indices = np.linspace(0, len(predictions), num_bins, endpoint=False)

  # Round into integers for indexing. If num_bins does not evenly divide
  # len(predictions), this means that bin sizes will alternate between SIZE and
  # SIZE+1.
  edge_indices = np.round(edge_indices).astype(int)

  # If there are many more bins than data points, some indices will be
  # out-of-bounds by one. Set them to be within bounds:
  edge_indices = np.minimum(edge_indices, len(predictions) - 1)

  # Obtain the edge values:
  edges = np.sort(predictions)[edge_indices]

  # Following the convention of numpy.digitize, we do not include the leftmost
  # edge (i.e. return the upper bin edges):
  return edges[1:]


def binary_converter(probs):
  """Converts a binary probability vector into a matrix."""
  return np.array([[1-p, p] for p in probs])


class GeneralCalibrationError():
  """Implements the space of calibration errors, General Calibration Error.

  This implementation of General Calibration Error can be class-conditional,
  adaptively binned, thresholded, focus on the maximum or top labels, and use
  the l1 or l2 norm. Can function as ECE, SCE, RMSCE, and more. For
  definitions of most of these terms, see [1].

  To implement Expected Calibration Error [2]:
  ECE = GeneralCalibrationError(binning_scheme='even', class_conditional=False,
    max_prob=True, error='l1')

  To implement Static Calibration Error [1]:
  SCE = GeneralCalibrationError(binning_scheme='even', class_conditional=False,
    max_prob=False, error='l1')

  To implement Root Mean Squared Calibration Error [3]:
  RMSCE = GeneralCalibrationError(binning_scheme='adaptive',
  class_conditional=True, max_prob=False, error='l2', datapoints_per_bin=100)

  To implement Adaptive Calibration Error [1]:
  ACE = GeneralCalibrationError(binning_scheme='adaptive',
  class_conditional=True, max_prob=False, error='l1')

  To implement Thresholded Adaptive Calibration Error [1]:
  TACE = GeneralCalibrationError(binning_scheme='adaptive',
  class_conditional=True, max_prob=False, error='l1', threshold=0.01)

  ### References

  [1] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
  and Dustin Tran. "Measuring Calibration in Deep Learning." In Proceedings of
  the IEEE Conference on Computer Vision and Pattern Recognition Workshops,
  pp. 38-41. 2019.
  https://arxiv.org/abs/1904.01685

  [2] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
  "Obtaining well calibrated probabilities using bayesian binning."
  Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/

  [3] Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich.
  "Deep anomaly detection with outlier exposure."
  arXiv preprint arXiv:1812.04606 (2018).
  https://arxiv.org/pdf/1812.04606.pdf

  Attributes:
    binning_scheme: String, either 'even' (for even spacing) or 'adaptive'
      (for an equal number of datapoints in each bin).

    max_prob: Boolean, 'True' to measure calibration only on the maximum
      prediction for each datapoint, 'False' to look at all predictions.

    class_conditional: Boolean, 'False' for the case where predictions from
      different classes are binned together, 'True' for binned separately.

    norm: String, apply 'l1' or 'l2' norm to the calibration error.

    num_bins: Integer, number of bins of confidence scores to use.

    threshold: Float, only look at probabilities above a certain value.

    datapoints_per_bin: Int, number of datapoints in each adaptive bin. This
      is a second option when binning adaptively - you can use either num_bins
      or this method to determine the bin size.

    distribution: String, data distribution this metric is measuring, whether
      train, test, out-of-distribution, or the user's choice.

    accuracies: Vector, accuracy within each bin.

    confidences: Vector, mean confidence within each bin.

    calibration_error: Float, computed calibration error.

    calibration_errors: Vector, difference between accuracies and confidences.
  """

  def __init__(self,
               binning_scheme,
               max_prob,
               class_conditional,
               norm,
               num_bins=30,
               threshold=0.0,
               datapoints_per_bin=None,
               distribution=None):
    self.binning_scheme = binning_scheme
    self.max_prob = max_prob
    self.class_conditional = class_conditional
    self.norm = norm
    self.num_bins = num_bins
    self.threshold = threshold
    self.datapoints_per_bin = datapoints_per_bin
    self.distribution = distribution
    self.accuracies = None
    self.confidences = None
    self.calibration_error = None
    self.calibration_errors = None

  def get_calibration_error(self, probs, labels, bin_upper_bounds, norm,
                            num_bins):
    """Given a binning scheme, returns sum weighted calibration error."""
    if np.size(probs) == 0:
      return 0.

    bin_indices = np.digitize(probs, bin_upper_bounds)
    sums = np.bincount(bin_indices, weights=probs, minlength=num_bins)
    sums = sums.astype(np.float64)  # In case all probs are 0/1.
    counts = np.bincount(bin_indices, minlength=num_bins)
    counts = counts + np.finfo(sums.dtype).eps  # Avoid division by zero.
    self.confidences = sums / counts
    self.accuracies = np.bincount(
        bin_indices, weights=labels, minlength=num_bins) / counts

    self.calibration_errors = self.accuracies-self.confidences

    weighting = counts / float(len(probs.flatten()))
    weighted_calibration_error = self.calibration_errors * weighting
    if norm == 'l1':
      return np.sum(np.abs(weighted_calibration_error))
    else:
      return np.sqrt(np.sum(np.square(weighted_calibration_error)))

  def update_state(self, labels, probs):
    """Updates the value of the General Calibration Error."""

    # if self.calibration_error is not None and

    probs = np.array(probs)
    labels = np.array(labels)
    if probs.ndim == 2:

      num_classes = probs.shape[1]
      if num_classes == 1:
        probs = probs[:, 0]
        probs = binary_converter(probs)
        num_classes = 2
    elif probs.ndim == 1:
      # Cover binary case
      probs = binary_converter(probs)
      num_classes = 2
    else:
      raise ValueError('Probs must have 1 or 2 dimensions.')

    # Convert the labels vector into a one-hot-encoded matrix.
    labels_matrix = one_hot_encode(labels, probs.shape[1])

    if self.datapoints_per_bin is not None:
      self.num_bins = int(len(probs)/self.datapoints_per_bin)
      if self.binning_scheme != 'adaptive':
        raise ValueError(
            "To set datapoints_per_bin, binning_scheme must be 'adaptive'.")

    if self.binning_scheme == 'even':
      bin_upper_bounds = np.histogram_bin_edges(
          [], bins=self.num_bins, range=(0.0, 1.0))[1:]

    # When class_conditional is False, different classes are conflated.
    if not self.class_conditional:
      if self.max_prob:
        labels_matrix = labels_matrix[
            range(len(probs)), np.argmax(probs, axis=1)]
        probs = probs[range(len(probs)), np.argmax(probs, axis=1)]
      labels_matrix = labels_matrix[probs > self.threshold]
      probs = probs[probs > self.threshold]
      if self.binning_scheme == 'adaptive':
        bin_upper_bounds = get_adaptive_bins(probs, self.num_bins)
      calibration_error = self.get_calibration_error(
          probs.flatten(), labels_matrix.flatten(), bin_upper_bounds, self.norm,
          self.num_bins)

    # If class_conditional is true, predictions from different classes are
    # binned separately.
    else:
      # Initialize list for class calibration errors.
      class_calibration_error_list = []
      for j in range(num_classes):
        if not self.max_prob:
          probs_slice = probs[:, j]
          labels = labels_matrix[:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          if self.binning_scheme == 'adaptive':
            bin_upper_bounds = get_adaptive_bins(probs_slice, self.num_bins)
          calibration_error = self.get_calibration_error(
              probs_slice, labels, bin_upper_bounds, self.norm, self.num_bins)
          class_calibration_error_list.append(calibration_error/num_classes)
        else:
          # In the case where we use all datapoints,
          # max label has to be applied before class splitting.
          labels = labels_matrix[np.argmax(probs, axis=1) == j][:, j]
          probs_slice = probs[np.argmax(probs, axis=1) == j][:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          if self.binning_scheme == 'adaptive':
            bin_upper_bounds = get_adaptive_bins(probs_slice, self.num_bins)
          calibration_error = self.get_calibration_error(
              probs_slice, labels, bin_upper_bounds, self.norm, self.num_bins)
          class_calibration_error_list.append(calibration_error/num_classes)
      calibration_error = np.sum(class_calibration_error_list)

    if self.norm == 'l2':
      calibration_error = np.sqrt(calibration_error)

    self.calibration_error = calibration_error

  def result(self):
    return self.calibration_error

  def reset_state(self):
    self.calibration_error = None


def gce(labels,
        probs,
        binning_scheme,
        max_prob,
        class_conditional,
        norm,
        num_bins=30,
        threshold=0.0,
        datapoints_per_bin=None):
  """Implements the space of calibration errors, General Calibration Error.

  This implementation of General Calibration Error can be class-conditional,
  adaptively binned, thresholded, focus on the maximum or top labels, and use
  the l1 or l2 norm. Can function as ECE, SCE, RMSCE, and more. For
  definitions of most of these terms, see [1].

  To implement Expected Calibration Error [2]:
  gce(labels, probs, binning_scheme='even', class_conditional=False,
    max_prob=True, error='l1')

  To implement Static Calibration Error [1]:
  gce(labels, probs, binning_scheme='even', class_conditional=False,
    max_prob=False, error='l1')

  To implement Root Mean Squared Calibration Error [3]:
  gce(labels, probs, binning_scheme='adaptive', class_conditional=True,
    max_prob=False, error='l2', datapoints_per_bin=100)

  To implement Adaptive Calibration Error [1]:
  gce(labels, probs, binning_scheme='adaptive', class_conditional=True,
    max_prob=False, error='l1')

  To implement Thresholded Adaptive Calibration Error [1]:
  gce(labels, probs, binning_scheme='adaptive', class_conditional=True,
    max_prob=False, error='l1', threshold=0.01)

  ### References

  [1] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
  and Dustin Tran. "Measuring Calibration in Deep Learning." In Proceedings of
  the IEEE Conference on Computer Vision and Pattern Recognition Workshops,
  pp. 38-41. 2019.
  https://arxiv.org/abs/1904.01685

  [2] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
  "Obtaining well calibrated probabilities using bayesian binning."
  Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/

  [3] Hendrycks, Dan, Mantas Mazeika, and Thomas Dietterich.
  "Deep anomaly detection with outlier exposure."
  arXiv preprint arXiv:1812.04606 (2018).
  https://arxiv.org/pdf/1812.04606.pdf

  Args:
    labels: np.ndarray of shape [N, ] array of correct labels.
    probs: np.ndarray of shape [N, M] where N is the number of datapoints
      and M is the number of predicted classes.
    binning_scheme: String, either 'even' (for even spacing) or 'adaptive'
      (for an equal number of datapoints in each bin).
    max_prob: Boolean, 'True' to measure calibration only on the maximum
      prediction for each datapoint, 'False' to look at all predictions.
    class_conditional: Boolean, 'False' for the case where predictions from
      different classes are binned together, 'True' for binned separately.
    norm: String, apply 'l1' or 'l2' norm to the calibration error.
    num_bins: Integer, number of bins of confidence scores to use.
    threshold: Float, only look at probabilities above a certain value.
    datapoints_per_bin: Int, number of datapoints in each adaptive bin. This
      is a second option when binning adaptively - you can use either num_bins
      or this method to determine the bin size.

  Raises:
    ValueError.

  Returns:
    Float, general calibration error.

  """
  metric = GeneralCalibrationError(num_bins=num_bins,
                                   binning_scheme=binning_scheme,
                                   class_conditional=class_conditional,
                                   max_prob=max_prob,
                                   norm=norm,
                                   threshold=threshold,
                                   datapoints_per_bin=datapoints_per_bin)
  metric.update_state(labels, probs)
  return metric.result()


general_calibration_error = gce


def ece(labels, probs, num_bins=30):
  """Implements Expected Calibration Error."""
  return gce(labels,
             probs,
             binning_scheme='even',
             max_prob=True,
             class_conditional=False,
             norm='l1',
             num_bins=num_bins)


def rmsce(labels, probs, num_bins=30, datapoints_per_bin=100):
  """Implements Root Mean Squared Calibration Error."""
  return gce(labels,
             probs,
             binning_scheme='adaptive',
             max_prob=True,
             class_conditional=False,
             norm='l2',
             num_bins=num_bins,
             datapoints_per_bin=datapoints_per_bin)

root_mean_squared_calibration_error = rmsce


def sce(labels, probs, num_bins=30):
  """Implements Static Calibration Error."""
  return gce(labels,
             probs,
             binning_scheme='even',
             max_prob=False,
             class_conditional=True,
             norm='l1',
             num_bins=num_bins)

static_calibration_error = sce


def ace(labels, probs, num_bins=30):
  """Implements Adaptive Calibration Error."""
  return gce(labels,
             probs,
             binning_scheme='adaptive',
             max_prob=False,
             class_conditional=True,
             norm='l1',
             num_bins=num_bins)

adaptive_calibration_error = ace


def tace(labels, probs, num_bins=30, threshold=0.01):
  """Implements Thresholded Adaptive Calibration Error."""
  return gce(labels,
             probs,
             binning_scheme='adaptive',
             max_prob=False,
             class_conditional=True,
             norm='l1',
             num_bins=num_bins,
             threshold=threshold)

thresholded_adaptive_calibration_error = tace


def compute_all_metrics(labels, probs):
  """Computes all GCE metrics."""
  parameters = [['even', 'adaptive'], [True, False], [True, False],
                [0.0, 0.01], ['l1', 'l2']]
  params = list(itertools.product(*parameters))
  measures = []
  for p in params:
    def metric(labels, probs, num_bins=30, p=p):
      """Implements Expected Calibration Error."""
      return gce(labels,
                 probs,
                 binning_scheme=p[0],
                 max_prob=p[1],
                 class_conditional=p[2],
                 threshold=p[3],
                 norm=p[4],
                 num_bins=num_bins)
    measures.append(metric(labels, probs))
  return np.array(measures)