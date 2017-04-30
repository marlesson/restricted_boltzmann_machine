require 'matrix'

class RBM
  
  attr_accessor :weights, :learning_rate

  # Initializes a new RBM.
  def initialize(num_visible, num_hidden, learning_rate = 0.1)

    @num_hidden    = num_hidden
    @num_visible   = num_visible
    @learning_rate = learning_rate

    init_weights
  end

  # samples = [
  #   [0, 1, 1, 1, 0, 0, ... ,n]
  #   [0, 1, 1, 1, 0, 0, ... ,n]
  #   [1, 1, 1, 0, 0, 0, ... ,n]
  #   [0, 1, 1, 0, 0, 1, ... ,n]
  #   [0, 1, 1, 1, 0, 0, ... ,n]
  #   [1, 0, 0, 0, 1, 1, ... ,n]
  # ]
  def train(samples, max_epochs = 5000)
    
    samples        = Marshal.load(Marshal.dump(samples)) # can't use dup since it's just a shallow copy
    num_examples   = samples.length

    # Insert bias units of 1 into the first column of each training_sample.
    samples.each { |sample| sample.unshift(1) }

    samples_matrix = Matrix[*samples]


    # Epochs of train
    max_epochs.times do |epoch|


      #### Step Positive
      # Uv = X . W
      pos_Uv     = samples_matrix * @weights

      # P (h1=1∣v )= sigmoide (c + wi v) 
      pos_G_Uv  = pos_Uv.map{|ij| logistic(ij)}


      # Pu(g(U v)) ={ pij ∣ pij =t }
      pos_P_G_Uv = pos_G_Uv.map{|ij|  (ij >= rand) ? 1.0 : 0.0}

      # A+
      pos_A = samples_matrix.transpose * pos_G_Uv


      #### Step Negative
      # Uv = X . W
      neg_Uh     = pos_P_G_Uv * @weights.transpose


      # P (h1=1∣v )= sigmoide (c + wi v) 
      # g (U¯h)
      neg_G_Uh  = neg_Uh.map{|ij| logistic(ij)}

      # Fix the bias unit to 1
      neg_G_Uh  = Matrix[*neg_G_Uh.to_a.map { |row| row[0] = 1; row }]


      # U¯v = g (U¯h) . W
      neg_Uv   = neg_G_Uh * @weights
        
      # g (U¯v)
      neg_G_Uv = neg_Uv.map { |e| logistic(e) }

      # A-
      neg_A = neg_G_Uh.transpose * neg_G_Uv


      # Update weights!
      @weights = @weights + @learning_rate*((pos_A - neg_A) / num_examples)

      error    = ((samples_matrix - neg_G_Uh).map { |e| e ** 2 }).reduce(:+)
      puts "Epoch #{ epoch }: error is #{ error }"
    end    
  end


  private

  def init_weights

    # Init wights with rando number
    @weights       = Array.new(@num_visible) { Array.new(@num_hidden) { num_rand } }
    
    # Insert weights for the bias units into the first row and first column.
    @weights.each { |row| row.unshift(0) }
    @weights.unshift(Array.new(@num_hidden + 1, 0))

    @weights = Matrix[*@weights]
  end

  def num_rand
    rand
  end

  def logistic(x)
    1.0 / (1.0 + Math.exp(-x))
  end
end