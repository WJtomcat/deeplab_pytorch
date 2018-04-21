




def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

def _split_divisible(num, num_ways, divisible_by=8):
  """Evenly splits num, num_ways so each piece is a multiple of divisible_by."""
  assert num % divisible_by == 0
  assert num / num_ways >= divisible_by
  # Note: want to round down, we adjust each split to match the total.
  base = num // num_ways // divisible_by * divisible_by
  result = []
  accumulated = 0
  for i in range(num_ways):
    r = base
    while accumulated + r < num * (i + 1) / num_ways:
      r += divisible_by
    result.append(r)
    accumulated += r
  assert accumulated == num
  return result

def expand_input_by_factor(n, divisible_by=8):
  return lambda num_inputs, **_: _make_divisible(num_inputs * n, divisible_by)
