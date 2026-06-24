# Model Fine-Tuning Results & Metrics

This document lists the detailed evaluation metrics, training logs, and class-by-class performance of the fine-tuned Vision Transformer (ViT-Base-Patch16-224) model on the Food-101 dataset.

---

## 1. Overall Metrics Summary

The model was trained for 5 epochs on a Tesla T4 GPU. The final metrics on the validation dataset are:

| Metric | Value |
| :--- | :--- |
| **Validation Accuracy** | **88.52%** |
| **Average Prediction Confidence** | **91.56%** |
| **Macro Average Precision** | **88.60%** |
| **Macro Average Recall** | **88.52%** |
| **Macro Average F1-Score** | **88.53%** |
| **Weighted Average Precision** | **88.60%** |
| **Weighted Average Recall** | **88.52%** |
| **Weighted Average F1-Score** | **88.53%** |
| **Best Class Accuracy** | **100.00%** (*edamame*) |
| **Worst Class Accuracy** | **62.80%** (*steak*) |
| **Training Loss** | **0.6951** (Average over 5 epochs) |
| **Training Time** | **92.19 minutes** (5,531.48 seconds) |

---

## 2. Epoch-by-Epoch Training Log

The progression of loss and accuracy across the 5 epochs shows steady convergence:

| Epoch | Training Loss | Validation Loss | Validation Accuracy (%) |
| :---: | :---: | :---: | :---: |
| 1 | 1.1423 | 0.8189 | 77.67% |
| 2 | 0.8003 | 0.6178 | 82.75% |
| 3 | 0.5317 | 0.5043 | 85.90% |
| 4 | 0.2487 | 0.4392 | 88.03% |
| 5 | 0.1260 | 0.4248 | **88.53%** (Best model loaded) |

---

## 3. Per-Class Performance Breakdown

### 🏆 Top 10 Best Performing Classes
These food types represent structured shapes, unique color templates, and low visual similarity to other items in the dataset, making them easily distinguishable by the self-attention blocks.

| Rank | Class | Accuracy | Correct Predictions / Total (250) |
| :---: | :--- | :---: | :---: |
| 1 | **edamame** | **100.00%** | 250 / 250 |
| 2 | **macarons** | **98.40%** | 246 / 250 |
| 3 | **hot_and_sour_soup** | **97.60%** | 244 / 250 |
| 4 | **spaghetti_bolognese** | **97.20%** | 243 / 250 |
| 5 | **oysters** | **96.40%** | 241 / 250 |
| 6 | **miso_soup** | **96.00%** | 240 / 250 |
| 7 | **guacamole** | **95.60%** | 239 / 250 |
| 8 | **takoyaki** | **95.60%** | 239 / 250 |
| 9 | **mussels** | **95.60%** | 239 / 250 |
| 10 | **eggs_benedict** | **95.60%** | 239 / 250 |

---

### ⚠️ Top 10 Worst Performing Classes
These food items represent classes with severe visual overlap. For example, meats look highly similar when cooked (e.g., *steak*, *pork_chop*, *filet_mignon*), and complex, multi-textured desserts share visual styles (e.g., *bread_pudding*, *apple_pie*, *chocolate_mousse*).

| Rank | Class | Accuracy | Correct Predictions / Total (250) |
| :---: | :--- | :---: | :---: |
| 1 | **steak** | **62.80%** | 157 / 250 |
| 2 | **pork_chop** | **72.00%** | 180 / 250 |
| 3 | **bread_pudding** | **73.60%** | 184 / 250 |
| 4 | **chocolate_mousse** | **74.40%** | 186 / 250 |
| 5 | **apple_pie** | **74.80%** | 187 / 250 |
| 6 | **filet_mignon** | **75.20%** | 188 / 250 |
| 7 | **cheesecake** | **77.20%** | 193 / 250 |
| 8 | **risotto** | **77.60%** | 194 / 250 |
| 9 | **foie_gras** | **77.60%** | 194 / 250 |
| 10 | **ravioli** | **77.60%** | 194 / 250 |

---

## 4. Full 101 Class-Wise Classification Report

The following table presents the precision, recall, and F1-score for each of the 101 food classes in the validation dataset (support = 250 images per class):

| Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| apple_pie | 0.7633 | 0.7480 | 0.7556 |
| baby_back_ribs | 0.9159 | 0.8280 | 0.8697 |
| baklava | 0.8769 | 0.9120 | 0.8941 |
| beef_carpaccio | 0.9109 | 0.9000 | 0.9054 |
| beef_tartare | 0.8838 | 0.8520 | 0.8676 |
| beet_salad | 0.8163 | 0.8000 | 0.8081 |
| beignets | 0.8966 | 0.9360 | 0.9159 |
| bibimbap | 0.9438 | 0.9400 | 0.9419 |
| bread_pudding | 0.7273 | 0.7360 | 0.7316 |
| breakfast_burrito | 0.8571 | 0.7920 | 0.8233 |
| bruschetta | 0.8504 | 0.7960 | 0.8223 |
| caesar_salad | 0.9055 | 0.9200 | 0.9127 |
| cannoli | 0.9409 | 0.9560 | 0.9484 |
| caprese_salad | 0.8577 | 0.9400 | 0.8969 |
| carrot_cake | 0.8780 | 0.8640 | 0.8710 |
| ceviche | 0.7761 | 0.8320 | 0.8031 |
| cheesecake | 0.8109 | 0.7720 | 0.7910 |
| cheese_plate | 0.9042 | 0.9440 | 0.9237 |
| chicken_curry | 0.8750 | 0.8400 | 0.8571 |
| chicken_quesadilla | 0.8810 | 0.8880 | 0.8845 |
| chicken_wings | 0.9323 | 0.9360 | 0.9341 |
| chocolate_cake | 0.7729 | 0.7760 | 0.7745 |
| chocolate_mousse | 0.7099 | 0.7440 | 0.7266 |
| churros | 0.9628 | 0.9320 | 0.9472 |
| clam_chowder | 0.8977 | 0.9480 | 0.9222 |
| club_sandwich | 0.9234 | 0.9160 | 0.9197 |
| crab_cakes | 0.8415 | 0.8280 | 0.8347 |
| creme_brulee | 0.9440 | 0.9440 | 0.9440 |
| croque_madame | 0.9315 | 0.9240 | 0.9277 |
| cup_cakes | 0.9137 | 0.9320 | 0.9228 |
| deviled_eggs | 0.9755 | 0.9560 | 0.9657 |
| donuts | 0.9102 | 0.8920 | 0.9010 |
| dumplings | 0.9036 | 0.9000 | 0.9018 |
| edamame | 0.9960 | 1.0000 | 0.9980 |
| eggs_benedict | 0.9409 | 0.9560 | 0.9484 |
| escargots | 0.8939 | 0.9440 | 0.9183 |
| falafel | 0.8566 | 0.8600 | 0.8583 |
| filet_mignon | 0.7402 | 0.7520 | 0.7460 |
| fish_and_chips | 0.9551 | 0.9360 | 0.9455 |
| foie_gras | 0.7293 | 0.7760 | 0.7519 |
| french_fries | 0.9714 | 0.9520 | 0.9616 |
| french_onion_soup | 0.9109 | 0.9000 | 0.9054 |
| french_toast | 0.8629 | 0.8560 | 0.8594 |
| fried_calamari | 0.8992 | 0.8920 | 0.8956 |
| fried_rice | 0.9163 | 0.9200 | 0.9182 |
| frozen_yogurt | 0.9595 | 0.9480 | 0.9537 |
| garlic_bread | 0.8941 | 0.8440 | 0.8683 |
| gnocchi | 0.8148 | 0.7920 | 0.8032 |
| greek_salad | 0.8712 | 0.9200 | 0.8949 |
| grilled_cheese_sandwich | 0.8154 | 0.8480 | 0.8314 |
| grilled_salmon | 0.8157 | 0.8320 | 0.8238 |
| guacamole | 0.9484 | 0.9560 | 0.9522 |
| gyoza | 0.8849 | 0.8920 | 0.8884 |
| hamburger | 0.8992 | 0.8560 | 0.8770 |
| hot_and_sour_soup | 0.9606 | 0.9760 | 0.9683 |
| hot_dog | 0.9237 | 0.9200 | 0.9218 |
| huevos_rancheros | 0.7903 | 0.8440 | 0.8162 |
| hummus | 0.9146 | 0.9000 | 0.9073 |
| ice_cream | 0.8686 | 0.8200 | 0.8436 |
| lasagna | 0.8904 | 0.8120 | 0.8494 |
| lobster_bisque | 0.9243 | 0.9280 | 0.9261 |
| lobster_roll_sandwich | 0.9240 | 0.9240 | 0.9240 |
| macaroni_and_cheese | 0.8893 | 0.8680 | 0.8785 |
| macarons | 0.9609 | 0.9840 | 0.9723 |
| miso_soup | 0.9756 | 0.9600 | 0.9677 |
| mussels | 0.9715 | 0.9560 | 0.9637 |
| nachos | 0.9198 | 0.8720 | 0.8953 |
| omelette | 0.8644 | 0.8160 | 0.8395 |
| onion_rings | 0.9151 | 0.9480 | 0.9312 |
| oysters | 0.9837 | 0.9640 | 0.9737 |
| pad_thai | 0.8868 | 0.9400 | 0.9126 |
| paella | 0.9124 | 0.9160 | 0.9142 |
| pancakes | 0.9105 | 0.9360 | 0.9231 |
| panna_cotta | 0.8405 | 0.8640 | 0.8521 |
| peking_duck | 0.9221 | 0.9000 | 0.9109 |
| pho | 0.9447 | 0.9560 | 0.9503 |
| pizza | 0.9478 | 0.9440 | 0.9459 |
| pork_chop | 0.7407 | 0.7200 | 0.7302 |
| poutine | 0.9289 | 0.9400 | 0.9344 |
| prime_rib | 0.8441 | 0.8880 | 0.8655 |
| pulled_pork_sandwich | 0.8519 | 0.9200 | 0.8846 |
| ramen | 0.9504 | 0.9200 | 0.9350 |
| ravioli | 0.7886 | 0.7760 | 0.7823 |
| red_velvet_cake | 0.8937 | 0.9080 | 0.9008 |
| risotto | 0.8326 | 0.7760 | 0.8033 |
| samosa | 0.9055 | 0.9200 | 0.9127 |
| sashimi | 0.9627 | 0.9280 | 0.9450 |
| scallops | 0.8340 | 0.8040 | 0.8187 |
| seaweed_salad | 0.9916 | 0.9480 | 0.9693 |
| shrimp_and_grits | 0.8029 | 0.8960 | 0.8469 |
| spaghetti_bolognese | 0.9567 | 0.9720 | 0.9643 |
| spaghetti_carbonara | 0.9832 | 0.9360 | 0.9590 |
| spring_rolls | 0.9454 | 0.9000 | 0.9221 |
| steak | 0.6133 | 0.6280 | 0.6206 |
| strawberry_shortcake | 0.8880 | 0.8880 | 0.8880 |
| sushi | 0.9012 | 0.9120 | 0.9066 |
| tacos | 0.8974 | 0.8400 | 0.8678 |
| takoyaki | 0.9336 | 0.9560 | 0.9447 |
| tiramisu | 0.8930 | 0.8680 | 0.8803 |
| tuna_tartare | 0.7445 | 0.8160 | 0.7786 |
| waffles | 0.8939 | 0.9440 | 0.9183 |
