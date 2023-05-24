from setuptools import setup, find_packages

setup(
    name="silicron",
    version="0.0.1",
    url="https://github.com/michaelliangau",
    author="Michael Liang",
    author_email="michaelliang15@gmail.com",
    description="Easily extend your chatbot with a knowledge base.",
    packages=find_packages(),
    install_requires=["fastapi"],
)
