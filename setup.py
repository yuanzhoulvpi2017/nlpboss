from distutils.core import setup
from setuptools import find_packages

with open("README.rst", "r", encoding='utf-8') as f:
    long_description = f.read()

setup(name='nlpboss',  # 包名
      version='1.0.0',  # 版本号
      description="a callback<send train-info to email> for HuggingFace's Transformers Trainer",
      long_description=long_description,
      author='yuanzhoulvpi2017',
      author_email='1582034172@qq.com',
      url='https://github.com/yuanzhoulvpi2017/nlpboss',
      install_requires=['pandas', 'transformers'],
      license='BSD License',
      packages=find_packages(),
      platforms=["all"],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Topic :: Software Development :: Libraries'
      ]
      )
