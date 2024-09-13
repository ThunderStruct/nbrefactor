xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
print(xs[-1])     # Negative indices count from the end of the list; prints "2"


xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)


xs.append('bar') # Add a new element to the end of the list
print(xs)


x = xs.pop()     # Remove and return the last element of the list
print(x, xs)

