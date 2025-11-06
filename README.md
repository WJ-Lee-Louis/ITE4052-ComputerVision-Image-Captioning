# ITE4052-ComputerVision-Image-Captioning
Image Captioning(COCO_captioning) Task with RNN, LSTM, Transformer

<img width="552" height="457" alt="image" src="https://github.com/user-attachments/assets/9335e897-c7a1-4a69-9bae-b594135fc40d" />
<img width="470" height="354" alt="image" src="https://github.com/user-attachments/assets/aaf0b0b0-1688-4a92-b347-7e634d361660" />
<img width="940" height="417" alt="image" src="https://github.com/user-attachments/assets/309c6e9a-2399-4b6c-83fc-a19362654c33" />



### Inference on NoCaps Test Dataset (feat. Hyperparameter Tuning)
Setting Scheduler: CosineAnnealing Learning Rate Scheduler


Cosine Annealing learning rate scheduler gradually decreases the learning rate from an initial maximum to a specified minimum by following the upper half of a cosine curve over the course of training. This smooth, non-linear decay allows larger weight updates early on, promoting rapid convergence, and finer adjustments later, helping the optimizer settle into a more optimal minimum. Optionally, incorporating warm restarts (Cosine Annealing with Restarts) resets the schedule periodically, enabling multiple cycles of exploration and refinement to further improve generalization.
