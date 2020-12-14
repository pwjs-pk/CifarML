import Augmentor
p = Augmentor.Pipeline("/path/to/source", output_directory="/path/to/output")
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
p.zoom(probability=0.3, min_factor=1.1, max_factor=1.6)
p.flip_top_bottom(0.3)
p.flip_left_right(0.3)
p.sample(300000)
