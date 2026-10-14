# Zero sanity baseline

Returns zero mean, covariance, and second moment using an empty factor matrix.
It ignores observations and edits deliberately. This is a loss/timing sanity
reference, not a useful estimator. Both modes use the same implementation and
CPU synchronization is a no-op.
