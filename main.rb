require "./rbm"

training_samples = [
  [0, 1, 1, 1, 0, 0],
  [0, 1, 1, 1, 0, 0],
  [1, 1, 1, 0, 0, 0],
  [0, 1, 1, 0, 0, 1],
  [0, 1, 1, 1, 0, 0],
  [1, 0, 0, 0, 1, 1],
]

rbm = RBM.new(6, 2)

rbm.train(training_samples)

puts rbm.weights.inspect