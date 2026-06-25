install:
	pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
	pip install -r requirements.txt

.PHONY: install