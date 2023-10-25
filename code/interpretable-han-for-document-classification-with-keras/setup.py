from setuptools import setup

setup(
    name='hierachical-attention-network-for-document-classification',
    version='0.1.0',
    packages=['han'],
    url='',
    description='An Keras inplementation of Hierarchical Attention Networks for document'
                ' classification.',
    install_requires=[
        'keras>=2.1.5',
        'pandas>=0.23.4'
    ]
)
