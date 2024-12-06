def binary_search(func, left, right, *, epsilon=1e-4):
  while right - left > epsilon:
    middle = (left + right) / 2
    if func(middle) > 0:
      right = middle
    else:
      left = middle
  return right