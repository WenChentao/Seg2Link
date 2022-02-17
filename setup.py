from setuptools import setup

setup(
    name='seg2link',
    version='0.1.0',
    packages=['seg2link'],
    entry_points={'console_scripts': ['seg2link=seg2link.start_seg2link:main'],},
    install_requires=[
            'napari[pyqt5]==0.4.10',
            'scikit-image==0.18.3'
        ],
    package_dir={'seg2link': 'seg2link'},
    url='https://github.com/WenChentao/Seg2Link',
    license='MIT',
    author='Chentao Wen',
    author_email='chintou.on@gmail.com',
    description='A napari based 3D segmentation software for electron microscopy images'
)
