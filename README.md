# Automatic Receipt OCR with PyTorch

This project uses Optical Character Recognition (OCR) technology to automatically extract text from receipts. The OCR functionality has been integrated into a local web app for easy access and usage. 

The OCR models used in this project include a receipt corners detector based on YOLOv4, a line detection model and a CNN-LSTM based OCR model, all  written in PyTorch.

## Getting Started

1. Clone this repository to your local machine

```git clone https://github.com/vantuan5644/ReceiptOCR.git```

2. Navigate to the project directory

```cd ReceiptOCR```

3. Install the necessary dependencies

```pip install -r requirements.txt```

4. Start the local webserver

```python server/app.py```

5. Access the web app by going to `localhost:5000` in your web browser

## References

The models used in this project were adapted from the following sources:

- Line detection model: [Zhou et al.’s 2017, EAST: An Efficient and Accurate Scene Text Detector.](https://arxiv.org/abs/1704.03155)

- OCR model: [Quoc 2020, VietOCR](https://github.com/pbcquoc/vietocr)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).


