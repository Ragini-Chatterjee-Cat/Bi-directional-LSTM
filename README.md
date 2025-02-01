
# This is a BiLSTM model that can be used to classify toxic comments without bias towards minority.
<p align="justify">
The BiLSTM model was built with several key components. The first was the embedding layer, which was initialized with pre-trained GloVe embeddings. I made sure to keep this layer non-trainable so that the model retained the pre-trained semantic knowledge. Following this, I used two stacked BiLSTM layers, each with 128 units, to ensure that both forward and backward context was captured. To condense the information from these layers, I applied both Global Max Pooling and Global Average Pooling. This helps in capturing both the most significant and the overall context from the sequences.. To further refine the learning process, I included dropout layers with rates between 0.2 and 0.5 to prevent overfitting, as well as batch normalization layers to stabilize training. The model had two final outputs: the primary output for toxicity classification and the auxiliary output for subgroup identification. 
</p>
<div align="center">
  <img width="370" alt="image" src="https://github.com/user-attachments/assets/e91a6319-49e8-4d7d-be4e-a9a7785a9cae" />
  <img width="387" alt="image" src="https://github.com/user-attachments/assets/7ab457d6-b86e-4e6e-8c91-7174f6b4959c" />
</div>
<p align="justify">
During training, I optimized the model using the Adam optimizer with a learning rate of 1e-3. The loss function used for both outputs was binary cross-entropy, but I assigned different weights to balance the importance of each. Since toxicity classification was the main goal, I set its loss weight to 1 while giving the auxiliary output a weight of 0.5. I trained the model using the training data and validation data for a total of 2 epochs with a batch size of 32.
</p>
