<h1>🔧 [1D classification] Fault Diagnosis using CNN</h1>

<p>This project implements a complete modular pipeline for classifying bearing faults using <strong>Convolutional Neural Networks</strong> on the <strong>CWRU</strong> (Case Western Reserve University) dataset.</p>

<hr/>

<h2>📁 Project Structure</h2>

<pre>
cnn_pipeline/
├── main.py                  # Main training/evaluation script
├── arguments.py             # Command-line arguments
├── dataset.py               # DataLoader generation
├── model.py                 # CNN model definition
├── train.py                 # Training function (with early stopping)
├── test.py                  # Test function
├── preprocess.py            # CWRU .mat file preprocessing
</pre>

<hr/>

<h2>📦 Data Structure</h2>

<pre>
raw_data/                    # Original CWRU .mat files
├── 97.mat                   # Normal
├── 105.mat                  # IR fault
├── 118.mat                  # Ball fault
├── 130.mat                  # Outer race fault
</pre>

<p>The <code>preprocess.py</code> module converts raw signals into sliding window segments for supervised classification.</p>

<hr/>

<h2>🚀 How to Run</h2>

<pre><code>pip install numpy scipy torch scikit-learn matplotlib tqdm</code></pre>

<pre><code>python main.py</code></pre>

<p>The script will:</p>
<ol>
  <li>Preprocess raw <code>.mat</code> files into (X, Y) arrays</li>
  <li>Split data into train/valid/test</li>
  <li>Train CNN using early stopping</li>
  <li>Evaluate final test accuracy</li>
</ol>

<hr/>

<h2>🧠 CNN Architecture</h2>

<p>Three-layer 1D convolutional encoder with ReLU, BatchNorm, and MaxPooling followed by fully connected classification layers.</p>

<pre>
Input → Conv1d → ReLU → MaxPool → Conv1d → ReLU → MaxPool → Conv1d → Flatten → FC → FC → Output
</pre>

<hr/>

<h2>📈 Training Options (config.py)</h2>

<table>
  <tr><th>Argument</th><th>Description</th><th>Default</th></tr>
  <tr><td><code>--epochs</code></td><td>Number of training epochs</td><td>10</td></tr>
  <tr><td><code>--lr</code></td><td>Learning rate</td><td>1e-4</td></tr>
  <tr><td><code>--lamda</code></td><td>LR scheduler decay</td><td>0.97</td></tr>
  <tr><td><code>--early_stop</code></td><td>Early stopping patience</td><td>20</td></tr>
  <tr><td><code>--train_size</code></td><td>Train/val split ratio</td><td>0.8</td></tr>
  <tr><td><code>--batch_size</code></td><td>Mini-batch size</td><td>64</td></tr>
</table>

<hr/>

