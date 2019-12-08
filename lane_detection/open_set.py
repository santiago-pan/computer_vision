import json


def open_set(path):
    with open(path + 'specs.json') as json_file:
        data = json.load(json_file)

        specs = {
            "frames": data['frames'],
            "base": data['base'],
            "ext": data['ext'],
            "zfill": data['zfill'],
            "tilt": data['tilt'],
            "far_point": data['far_point'],
            "near_point": data['near_point'],
            "center_point": data['center_point'],
            "far_aperture": data['far_aperture'],
            "near_aperture": data['near_aperture'],
            "fps": data['fps']
        }

        return specs
