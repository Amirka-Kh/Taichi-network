import taichi as ti
import mnist

# Initialization
ti.init(arch=ti.cpu, default_fp=ti.f32)  # debug=True

# Data type shortcuts
real = ti.f32
scalar = lambda: ti.var(dt=real)

# Number of recognized digits
n_numbers = 10

# Image size
n_pixels = 28 ** 2
image_size = 28

# POLICY HYPERPARAMETERS
N_HIDDEN = 500

# TRAINING HYPERPARAMETERS
TRAINING_EPOCHS = 5

# Data types
pixels = scalar()
weights1 = scalar()
output1 = scalar()
weights2 = scalar()
output2 = scalar()
output2_exp = scalar()
output2_norm = scalar()
needed = scalar()
output_sum = scalar()
loss = scalar()
learning_rate = scalar()


# Data layout configuration (for faster computation)
@ti.layout
def place():
    ti.root.dense(ti.i, n_pixels).place(pixels)
    ti.root.dense(ti.ij, (n_pixels, N_HIDDEN)).place(weights1)
    ti.root.dense(ti.i, N_HIDDEN).place(output1)
    ti.root.dense(ti.ij, (N_HIDDEN, n_numbers)).place(weights2)
    ti.root.dense(ti.i, n_numbers).place(output2)
    ti.root.dense(ti.i, n_numbers).place(output2_exp)
    ti.root.dense(ti.i, n_numbers).place(output2_norm)
    ti.root.dense(ti.i, n_numbers).place(needed)
    ti.root.place(output_sum)
    ti.root.place(loss)
    ti.root.place(learning_rate)

    # Add gradient variables
    ti.root.lazy_grad()


# Initialize network
@ti.kernel
def init_weights_biases():
    # Layer 1
    for i in range(n_pixels):
        for j in range(N_HIDDEN):
            weights1[i, j] = ti.random() * 0.005

    # Layer 2
    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            weights2[i, j] = ti.random() * 0.005


# Clear gradients and outputs
@ti.kernel
def clear_weights_biases_grad():
    # Layer 1
    for i in range(n_pixels):
        for j in range(N_HIDDEN):
            weights1.grad[i, j] = 0

    # Layer 2
    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            weights2.grad[i, j] = 0


def clear_outputs_grad():
    # Layer 1
    for i in range(N_HIDDEN):
        output1[i] = 0
        output1.grad[i] = 0

    # Layer 2
    for i in range(n_numbers):
        output2[i] = 0
        output2.grad[i] = 0
        output2_exp[i] = 0
        output2_exp.grad[i] = 0
        output2_norm[i] = 0
        output2_norm.grad[i] = 0


# Compute layers
@ti.kernel
def layer1():
    for i in range(n_pixels):
        for j in range(N_HIDDEN):
            output1[j] += pixels[i] * weights1[i, j]

    for i in range(N_HIDDEN):
        output1[i] = ti.tanh(output1[i])


@ti.kernel
def layer2():
    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            output2[j] += output1[i] * weights2[i, j]

    for i in range(n_numbers):
        output2_exp[i] = ti.exp(output2[i])
        output_sum[None] += output2_exp[i] + 1e-6

    # Normalization
    for i in range(n_numbers):
        output2_norm[i] = output2_exp[i] / output_sum[None]


# Compute loss (cross-entropy)
@ti.kernel
def compute_loss():
    for i in range(n_numbers):
        loss[None] += (-needed[i]) * ti.log(output2_norm[i])


# Gradient descent
@ti.kernel
def gd_layer1():
    for i in range(n_pixels):
        for j in range(N_HIDDEN):
            weights1[i, j] -= learning_rate * weights1.grad[i, j]


@ti.kernel
def gd_layer2():
    for i in range(N_HIDDEN):
        for j in range(n_numbers):
            weights2[i, j] -= learning_rate * weights2.grad[i, j]


# Step forward through network
def forward():
    layer1()
    layer2()
    compute_loss()


# Step back to compute gradients
def backward_grad():
    compute_loss.grad()
    layer2.grad()
    layer1.grad()


# MNIST images
training_images = mnist.train_images()
training_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Initialize network
init_weights_biases()


# Compute accuracy of predictions on tests
def test_accuracy():
    n_test = len(test_images) // 10
    accuracy = 0
    for i in range(n_test):
        # Input
        curr_image = test_images[i]
        for j in range(image_size):
            for k in range(image_size):
                pixels[image_size * j + k] = curr_image[j][k] / 255
        for j in range(n_numbers):
            needed[j] = int(test_labels[i] == j)

        clear_outputs_grad()
        clear_weights_biases_grad()
        loss[None] = 0

        forward()

        outputs = []
        for j in range(n_numbers):
            outputs.append(output2[j])
        prediction = outputs.index(max(outputs))  # Digit with higher prediction
        accuracy += int(prediction == test_labels[i])

    return accuracy / n_test


# Training
def main():
    # Training
    losses = []
    accuracies = []
    for n in range(TRAINING_EPOCHS):
        for i in range(len(training_images)):
            learning_rate[None] = 5e-3 * (0.1 ** (2 * i // 60000))

            # Input
            curr_image = training_images[i]
            for j in range(image_size):
                for k in range(image_size):
                    pixels[image_size * j + k] = curr_image[j][k] / 255
            for j in range(n_numbers):
                needed[j] = int(training_labels[i] == j)

            clear_outputs_grad()
            clear_weights_biases_grad()
            output_sum[None] = 0
            loss[None] = 0

            forward()

            curr_loss = loss[None]
            losses.append(curr_loss)
            losses = losses[-100:]
            if i % 100 == 0:
                print('i =', i, ' loss : ', sum(losses) / len(losses))
            if i % 1000 == 0:
                curr_acc = test_accuracy()
                print('test accuracy: {:.2f}%'.format(100 * curr_acc))
                accuracies.append(curr_acc)

            loss.grad[None] = 1
            output_sum.grad[None] = 0

            backward_grad()

            gd_layer1()
            gd_layer2()


if __name__ == '__main__':
    main()
