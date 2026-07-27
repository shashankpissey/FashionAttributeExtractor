import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_ciede2000

from utils.log_config import get_logger
logger = get_logger(__name__)
class ColourTagger:

    def __init__(self, color_names):
        """
        below colour centroids are derived from sRGB centroid paper. The colour is matched to its nearest colour from the list provided in the paper
        """
        self.colours = {
            'white': (231, 225, 233),
            'black': (43, 41, 43),
            'grey': (138, 132, 137),
            'light grey':(189, 183, 191),
            'dark grey': (88, 84, 88),
            'silver': (189, 183, 191),
            'red': (191, 52, 75),
            'burgundy': (136, 16, 85),
            'maroon': (116, 36, 52),
            'blue': (39, 108, 189),
            'light blue': (115, 164, 220),
            'dark blue': (23, 52, 89),
            'navy': (30, 37, 49),
            'turquoise': (19, 133, 175),
            'teal': (13, 143, 130),
            'green': (21, 138, 102),
            'light green': (111, 172, 149),
            'dark green': (22, 78, 61),
            'sage green': (97, 113, 110),
            'olive green':  (73, 91, 34),
            'lime': (167, 220, 38),
            'khaki': (169, 144, 102),
            'yellow': (217, 180, 81),
            'light yellow': (244, 210, 132),
            'gold': (217, 174, 47),
            'orange': (234, 129, 39),
            'rust': (175, 51, 24),
            'pink': (244, 143, 160),
            'peach': (251, 175, 130),
            'purple': (147, 82, 168),
            'lavender': (155, 140, 202),
            'brown': (127, 72, 41),
            'dark brown': (68, 33, 18),
            'light brown': (173, 124, 99),
            'tan': (196, 154, 116),
            'beige': (198, 185, 177),
            'cream': (238, 223, 218),
            'clear': (231, 225, 233)
            }
        self.DEFAULT_COLOUR = {
            "clear": (255, 255, 255)
            }
        self.lab_colour_names, self.lab_values = self.build_main_colours(color_names)
        
    def get_lab_from_rgb(self, rgb):
        """
        From RGB obtain the LAB values and also normalise it as it is required for comparing the distances in CIE
        """
        rgb_array = np.array([[rgb]], dtype=np.float64) / 255.0
        return rgb2lab(rgb_array)[0][0]
    
    def build_main_colours(self, colour_names):
        """
        Builds LAB reference values from the manually defined ISCC-NBS-based
        RGB anchors (self.colours)
        if a colour name that is  found in self.colours then it defaults it to clear.
        """
        colour_name_list, lab_values = [], []

        for colour in colour_names:
            if colour in self.colours:
                rgb = self.colours[colour]
            else:
                logger.warning(f"{colour} was not found in manual anchors and hence being defaulted to white")
                rgb = self.DEFAULT_COLOUR.get("clear", (255, 255, 255))

            colour_name_list.append(colour)
            lab_values.append(self.get_lab_from_rgb(rgb))

        return colour_name_list, np.array(lab_values)
    
    def extract_dominant_colour(self, image, colour_names, lab_values, num_k):
        """
        This method reads the mask and runs a Kmeans cluster, obtains the dominant colour and return it along with the percentage of it occupying within mask
        """

        image = np.array(image.convert("RGBA"))
        arr = np.array(image)

        rgb = arr[:, :, :3].reshape(-1, 3).astype(np.float32)
        alpha = arr[:, :, 3].reshape(-1)

        mask = alpha == 255
        pixels = rgb[mask]

        if len(pixels) == 0:
            logger.warning("No fully opaque pixels found — falling back to alpha > 0")
            pixels = rgb[alpha > 0]

        pixels_rgb_norm = pixels.reshape(-1, 1, 3).astype(np.float64) / 255.0
        # Convert to LAB value
        pixels_lab = rgb2lab(pixels_rgb_norm).reshape(-1, 3)

        kmeans = KMeans(n_clusters = num_k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(pixels_lab)
        counts = np.bincount(kmeans.labels_, minlength=num_k)
        dominant_colour = kmeans.cluster_centers_[np.argmax(counts)]
        percentage = counts[np.argmax(counts)]/ len(pixels)

        distances = deltaE_ciede2000(dominant_colour.reshape(-1, 3), lab_values)
        nearest_colour_index = int(np.argmin(distances))

        colour_dict = {
            "predicted_colour": colour_names[nearest_colour_index],
            "percentage": percentage,
            "delta_e_distance": float(distances[nearest_colour_index])
        }

        return colour_dict
    
    def extract_colours(self, image, k):
        # Main methof that runs the code to extract the dominant colour

        result = self.extract_dominant_colour(image, self.lab_colour_names, self.lab_values, k)

        return result

