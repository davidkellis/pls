# pls
A Scala implementation of the PLS algorithms described in Dayal and MacGregor's "Improved PLS Algorithms" paper,
published in Journal of Chemometrics, Volume 11, Issue 1, pages 73–85, January 1997.

## Sample data sets

Some sample data sets are in ./data:

1. artificial13.csv was produced with the following ruby snippet:
  ```
  coefficients = -5..7   # 13 coefficients between [-5, -4, ..., 6, 7]
  200.times.map do
  observation_vector = 13.times.map{ rand(1000).to_f }
  y = coefficients.zip(observation_vector).map{|pair| coeff, observation = *pair; coeff * observation }.reduce(:+)
  puts ([y] + observation_vector).join(",")
  end; nil
  ```
  So the data conforms to the equation:

  <code>
  y = -5*a + -4*b + -3*c + -2*d + -1*e + 0*f + 1*g + 2*h + 3*i + 4*j + 5*k + 6*l
  </code>

2. artificial100.csv was produced with the following ruby snippet:
  ```
  coefficients = -50...50   # 100 coefficients between [-50, -49, -48, ..., 48, 49]
  200.times.map do
  observation_vector = coefficients.count.times.map{ rand(100000).to_f }
  y = coefficients.zip(observation_vector).map{|pair| coeff, observation = *pair; coeff * observation }.reduce(:+)
  puts ([y] + observation_vector).join(",")
  end; nil
  ```
  So the data conforms to the equation:

  <code>
  y = -50*x_1 + -49*x_2 + ... + 48*x_99 + 49*x_100
  </code>



coefficients = -5000...5000   # 10,000 coefficients between [-5000, 4999]
open("artificial#{coefficients.count}.csv", 'w') do |f|
  20_000.times.map do
    observation_vector = coefficients.count.times.map{ rand(100000).to_f }
    y = coefficients.zip(observation_vector).map{|pair| coeff, observation = *pair; coeff * observation }.reduce(:+)
    f.puts ([y] + observation_vector).join(",")
  end
end
