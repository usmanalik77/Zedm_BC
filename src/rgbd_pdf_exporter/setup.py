from setuptools import find_packages, setup

package_name = 'rgbd_pdf_exporter'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'matplotlib', 'reportlab'],
    zip_safe=True,
    maintainer='usman',
    maintainer_email='usman@todo.todo',
    description='Package to export RGB-D images to PDF',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rgbd_pdf_exporter = rgbd_pdf_exporter.rgbd_pdf_exporter:main',
        ],
    },
)
