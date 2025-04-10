<h1>🔧 Fault Diagnosis using WGAN-CGAN on CWRU Dataset</h1>

<p>
This project implements a <strong>Wasserstein Conditional GAN (WGAN-CGAN)</strong> model for generating class-conditional vibration signals using the <strong>CWRU (Case Western Reserve University)</strong> dataset. It aims to synthesize realistic 1D signals corresponding to different bearing fault types and compare their spectral similarity to the real signals.
</p>

<hr/>

<h2>📁 Project Structure</h2>

<pre>
GAN_CWRU/
├── main.py                  # Main training/evaluation script
├── arguments.py             # CLI training options
├── model.py                 # WGAN-CGAN Generator & Discriminator
├── train.py                 # WGAN-CGAN training loop
├── test.py                  # FFT-based evaluation & plotting
├── dataset.py               # DataLoader construction
├── preprocess.py            # CWRU .mat preprocessing to (X, Y)
</pre>

<hr/>

<h2>📦 Data Structure</h2>

<pre>
raw_data/
├── 97.mat     # Normal
├── 105.mat    # Inner Race fault
├── 118.mat    # Ball fault
├── 130.mat    # Outer Race fault
</pre>

<p>Each file contains DE-side vibration signals under different operating conditions.</p>

<p><strong>Preprocessing</strong> extracts 4000 samples per class using a sliding window of length 1200 points.</p>

<hr/>

<h2>🚀 How to Run</h2>

<pre><code>pip install numpy torch scipy matplotlib pandas scikit-learn tqdm</code></pre>

<pre><code>python main.py</code></pre>

<p>The training script will:</p>
<ol>
  <li>Load and preprocess the CWRU dataset</li>
  <li>Train the WGAN-CGAN model to generate fault-specific signals</li>
  <li>Evaluate spectral similarity (FFT cosine similarity) between real and generated samples</li>
  <li>Visualize results with boxplots and FFT curves</li>
</ol>

<hr/>

<h2>🧠 Model Overview</h2>

<ul>
  <li><strong>Generator</strong>: MLP that takes noise + class label → vibration signal</li>
  <li><strong>Discriminator</strong>: MLP that scores real/fake given signal + class label</li>
  <li><strong>Loss</strong>: Wasserstein loss (with weight clipping)</li>
  <li><strong>Optimization</strong>: RMSProp (as recommended in WGAN)</li>
</ul>

<hr/>

<h2>📊 Evaluation Method</h2>

<p>
Generated samples are compared with real signals from each class using <strong>FFT-based cosine similarity</strong>.<br/>
For each class, the average FFT similarity is computed and plotted:
</p>

<ul>
  <li>📈 <code>fft_similarity_boxplot.png</code>: Boxplot per class</li>
  <li>🎨 <code>fft_comparison_samples.png</code>: Real vs Generated FFT curves</li>
</ul>

<hr/>

<h2>⚠️ Observations</h2>

<ul>
  <li>Class-conditional generation works well for most fault types (IR, Ball, Outer).</li>
  <li><strong>Normal data</strong> is harder to generate realistically due to lack of high-frequency components or variation.</li>
</ul>

<p>
Future improvements (e.g., <strong>spectral loss</strong>, <strong>multi-resolution FFTs</strong>, or <strong>1D CNN-based discriminator</strong>) could enhance the ability to capture subtle patterns in normal signals.
</p>

<hr/>

<h2>📎 Reference</h2>

<ul>
  <li>🔗 CWRU Dataset: <a href="https://engineering.case.edu/bearingdatacenter/download-data-file" target="_blank">https://engineering.case.edu/bearingdatacenter/download-data-file</a></li>
  <li>📄 WGAN Paper: Arjovsky et al., 2017. "Wasserstein GAN"</li>
  <li>📄 CGAN Paper: Mirza & Osindero, 2014. "Conditional GAN"</li>
</ul>
