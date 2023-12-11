There are two versions of our code to run: The Genetic Algorithm code and the Gradient Descent code.

The first step is to clone the repository using `git clone`.

Next, move into the finalProject directory using `cd finalProject`.

Now, to run the gradient descent code, run `cargo run --release` for optimized performance. Note, this code will take a super long time to run (we left it overnight and it still didn't finish), but you can see the backpropogation in action through the command line.

In order to run the Genetic Algorithm code, navigate into the branch titled "Genetic-Algorithm" using `git checkout 'Genetic-Algorithm'`.

Then, run `cargo run --release` again.  This code takes less time then the gradient descent so it should finish in less than 10ish minutes, and you can see the losses decreasing through a print statement of the accumulated loss of the best neuron in each generation.
