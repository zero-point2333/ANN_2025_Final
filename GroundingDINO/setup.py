import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="groundingdino",
    version="0.1.0",
    author="Yue Xu, Yichen Cai, Jinyu Yang",
    author_email="",
    description="Grounding DINO by Jittor",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IDEA-Research/Grounding-DINO",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "jittor==1.3.10",
        "addict==2.4.0",
        "yapf==0.43.0",
        "numpy==1.21.6",
        "opencv-python==4.5.5.64",
        "supervision==0.25.1",
        "pycocotools==2.0.7",
        "transformers==4.46.3",
        "torch --index-url https://download.pytorch.org/whl/cu121",
        "torchvision --index-url https://download.pytorch.org/whl/cu121",
    ],
)
