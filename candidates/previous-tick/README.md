# Previous-tick baseline

Maintains a point table with replacement and deletion semantics. Each prediction
rescans the table, using a historical query's window or the `lookback` interval
ending at `asof` for forecasts. The most recent observation before the window may
anchor its first grid value. Only observations at or before the permitted window
end are used; grid values carry the latest known point forward without future
interpolation.

`bins` is the number of equally spaced grid points, producing `bins - 1`
returns. Per-asset mean returns estimate a drift rate. Centered returns estimate
a covariance rate, represented by factor columns without a dense matrix. Returns
before an asset's first known point are omitted; their centered factor entries
are zero, so partial overlap can attenuate cross-covariance. Assets with fewer
than two usable points have zero estimated rates. The fixed `floor` adds diagonal
covariance rate explicitly; it is not estimated observation noise.

Mean and covariance rates are multiplied by the requested interval duration.
The evaluator derives the second moment by adding the mean outer product. This
simple baseline assumes stable rates over its estimation window and forecast
horizon, and does not correct observation noise. Online edits are inexpensive
table operations, while window rescanning and grid construction are charged to
prediction. CPU synchronization is a no-op.
