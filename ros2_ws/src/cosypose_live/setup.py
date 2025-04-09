from setuptools import find_packages, setup

package_name = 'cosypose_live'


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='magnus',
    maintainer_email='magnu.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "webcam = cosypose_live.webcam_publisher:main",
            "vision = cosypose_live.webcam_subscriber:main",
            "detection = cosypose_live.detection_node:main"
        ],
    },
)

