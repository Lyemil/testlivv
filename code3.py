import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import json  # Add this line to import the JSON module
from typing import Dict
from pyzbar.pyzbar import decode

# Function to load biomarker regions from JSON
def load_biomarker_regions(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find file at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

# Load the biomarker regions data with correct path
biomarker_regions = load_biomarker_regions('biomarker_regions.json')  # Assuming the file is in the same directory

if biomarker_regions is None:
    print("Failed to load biomarker regions data. Please check the file path and format.")
    exit(1)

# Color ranges definitions
COLOR_RANGES = {
    # Orange variations
    'orange': (np.array([5, 100, 100], dtype="uint8"), np.array([15, 255, 255], dtype="uint8")),
    'orange_dark': (np.array([5, 70, 70], dtype="uint8"), np.array([15, 255, 200], dtype="uint8")),
    'orange_light': (np.array([5, 50, 150], dtype="uint8"), np.array([15, 150, 255], dtype="uint8")),
    
    # Cyan variations
    'cyan': (np.array([85, 80, 80], dtype="uint8"), np.array([135, 255, 255], dtype="uint8")),
    'cyan_dark': (np.array([85, 60, 60], dtype="uint8"), np.array([135, 255, 200], dtype="uint8")),
    'cyan_light': (np.array([85, 40, 120], dtype="uint8"), np.array([135, 180, 255], dtype="uint8")),
    
    # Purple variations
    'purple': (np.array([130, 50, 50], dtype="uint8"), np.array([160, 255, 255], dtype="uint8")),
    'purple_dark': (np.array([130, 30, 30], dtype="uint8"), np.array([160, 200, 200], dtype="uint8")),
    'purple_light': (np.array([130, 30, 120], dtype="uint8"), np.array([160, 150, 255], dtype="uint8")),
    
    # Beige variations
    'beige': (np.array([20, 20, 180], dtype="uint8"), np.array([30, 50, 255], dtype="uint8")),
    'beige_dark': (np.array([20, 15, 140], dtype="uint8"), np.array([30, 45, 220], dtype="uint8")),
    'beige_light': (np.array([20, 10, 200], dtype="uint8"), np.array([30, 40, 255], dtype="uint8")),
    'beige_medium': (np.array([15, 30, 130], dtype="uint8"), np.array([25, 60, 200], dtype="uint8")),
    'beige_rose': (np.array([10, 25, 130], dtype="uint8"), np.array([20, 55, 200], dtype="uint8")),
    'beige_warm': (np.array([12, 28, 125], dtype="uint8"), np.array([22, 58, 195], dtype="uint8")),
    'beige_cool': (np.array([18, 22, 135], dtype="uint8"), np.array([28, 52, 205], dtype="uint8")),
    
    # Green variations
    'dark_green': (np.array([40, 40, 40], dtype="uint8"), np.array([80, 255, 255], dtype="uint8")),
    'green_dark': (np.array([40, 30, 30], dtype="uint8"), np.array([80, 200, 200], dtype="uint8")),
    'green_light': (np.array([40, 25, 100], dtype="uint8"), np.array([80, 180, 255], dtype="uint8")),
    
    # Yellow variations
    'yellow': (np.array([17, 38, 169], dtype="uint8"), np.array([30, 255, 255], dtype="uint8")),
    'yellow_dark': (np.array([17, 30, 130], dtype="uint8"), np.array([30, 200, 220], dtype="uint8")),
    'yellow_2': (np.array([31, 100, 100], dtype="uint8"), np.array([35, 255, 255], dtype="uint8")),
    'yellow_pale': (np.array([25, 25, 150], dtype="uint8"), np.array([35, 150, 255], dtype="uint8")),
    
    # Pink variations
    'light_pink': (np.array([160, 30, 180], dtype="uint8"), np.array([180, 100, 255], dtype="uint8")),
    'pink_dark': (np.array([160, 25, 140], dtype="uint8"), np.array([180, 90, 220], dtype="uint8")),
    'pink_pale': (np.array([160, 20, 200], dtype="uint8"), np.array([180, 80, 255], dtype="uint8")),
    
    # Rose variations
    'rose': (np.array([0, 30, 180], dtype="uint8"), np.array([10, 100, 255], dtype="uint8")),
    'rose_dark': (np.array([0, 25, 140], dtype="uint8"), np.array([10, 90, 220], dtype="uint8")),
    'rose_light': (np.array([0, 20, 200], dtype="uint8"), np.array([10, 80, 255], dtype="uint8")),
    
    # Tan variations
    'tan': (np.array([15, 20, 150], dtype="uint8"), np.array([25, 100, 255], dtype="uint8")),
    'tan_dark': (np.array([15, 15, 120], dtype="uint8"), np.array([25, 80, 220], dtype="uint8")),
    'tan_light': (np.array([15, 10, 180], dtype="uint8"), np.array([25, 70, 255], dtype="uint8")),
    'tan_rose': (np.array([12, 18, 125], dtype="uint8"), np.array([22, 75, 215], dtype="uint8")),
    'tan_warm': (np.array([14, 22, 130], dtype="uint8"), np.array([24, 82, 210], dtype="uint8")),
    
    # White variations
    'white_1': (np.array([0, 0, 160], dtype="uint8"), np.array([180, 50, 255], dtype="uint8")),
    'white_2': (np.array([50, 0, 160], dtype="uint8"), np.array([80, 50, 255], dtype="uint8")),
    'white_3': (np.array([20, 0, 160], dtype="uint8"), np.array([40, 50, 255], dtype="uint8")),
    'white_4': (np.array([60, 0, 160], dtype="uint8"), np.array([100, 50, 255], dtype="uint8")),
    'white_low': (np.array([0, 0, 140], dtype="uint8"), np.array([180, 40, 220], dtype="uint8")),
    
    # Brown variations
    'brown': (np.array([10, 50, 50], dtype="uint8"), np.array([20, 255, 200], dtype="uint8")),
    'brown_dark': (np.array([10, 30, 30], dtype="uint8"), np.array([20, 200, 150], dtype="uint8")),
    
    # Gray variations
    'gray': (np.array([0, 0, 100], dtype="uint8"), np.array([180, 30, 200], dtype="uint8")),
    'gray_dark': (np.array([0, 0, 60], dtype="uint8"), np.array([180, 25, 150], dtype="uint8")),
    
    # Beige-Rose variations
    'beige_rose_light': (np.array([8, 15, 140], dtype="uint8"), np.array([18, 45, 220], dtype="uint8")),
    'beige_rose_dark': (np.array([5, 20, 120], dtype="uint8"), np.array([15, 50, 200], dtype="uint8")),
    'beige_rose_warm': (np.array([7, 18, 130], dtype="uint8"), np.array([17, 48, 210], dtype="uint8")),
    
    # Pink-Beige variations
    'pink_beige': (np.array([0, 15, 150], dtype="uint8"), np.array([10, 45, 230], dtype="uint8")),
    'pink_beige_light': (np.array([0, 10, 160], dtype="uint8"), np.array([10, 40, 240], dtype="uint8")),
    'pink_beige_dark': (np.array([0, 20, 140], dtype="uint8"), np.array([10, 50, 220], dtype="uint8")),
}

# Update the color priority dictionary to include new colors
color_priority = {
    'yellow': 1, 'yellow_dark': 2, 'yellow_2': 3, 'yellow_pale': 4,
    'orange': 5, 'orange_dark': 6, 'orange_light': 7,
    'purple': 8, 'purple_dark': 9, 'purple_light': 10,
    'cyan': 11, 'cyan_dark': 12, 'cyan_light': 13,
    'dark_green': 14, 'green_dark': 15, 'green_light': 16,
    'beige': 17, 'beige_dark': 18, 'beige_light': 19,
    'beige_medium': 20,
    'beige_rose': 21,
    'beige_warm': 22,
    'beige_cool': 23,
    'tan': 24, 'tan_dark': 25, 'tan_light': 26,
    'tan_rose': 27,
    'tan_warm': 28,
    'white_1': 29, 'white_2': 30, 'white_3': 31, 'white_4': 32, 'white_low': 33,
    'brown': 34, 'brown_dark': 35,
    'gray': 36, 'gray_dark': 37,
    'beige_rose_light': 21,
    'beige_rose_dark': 22,
    'beige_rose_warm': 23,
    'pink_beige': 24,
    'pink_beige_light': 25,
    'pink_beige_dark': 26,
}

# Add these new static positions and test interpretation functions
static_positions = {
    'left_column': ['KET', 'PRO', 'pH', 'SG', 'NA'],  # sodium as NA
    'right_column': ['MG', 'CA', 'MDA', 'CRE', 'VC']  # Magnesium, Calcium, MDA, Creatinine, Vitamin C
}

def interpret_test_results(region):
    """Interpret test results based on color and position"""
    test_name = region.get('test_name', '')
    rgb = region['rgb']
    
    result = {
        'value': None,
        'level': None,
        'description': None
    }

    def color_distance(c1, c2):
        """Calculate color distance between two RGB values"""
        # Ensure we're working with single RGB values, not lists of RGB values
        if isinstance(c2, list) and isinstance(c2[0], list):
            # If c2 is a list of RGB values, find the closest one
            return min(color_distance(c1, single_rgb) for single_rgb in c2)
        
        # Enhanced color distance calculation using weighted RGB components
        r_weight = 0.3
        g_weight = 0.59
        b_weight = 0.11
        
        try:
            r_diff = (c1[0] - c2[0]) * r_weight
            g_diff = (c1[1] - c2[1]) * g_weight
            b_diff = (c1[2] - c2[2]) * b_weight
            return (r_diff**2 + g_diff**2 + b_diff**2) ** 0.5
        except Exception as e:
            print(f"Error calculating color distance: {e}")
            print(f"c1: {c1}, c2: {c2}")
            return float('inf')  # Return infinity for invalid comparisons

    test_name_map = {
        'KET': 'Ketone',
        'PRO': 'Protein',
        'pH': 'pH',
        'SG': 'Specific_Gravity',
        'NA': 'Sodium',
        'MG': 'Magnesium',
        'CA': 'Calcium',
        'MDA': 'MDA',
        'CRE': 'Creatinine',
        'VC': 'Vitamin_C'
    }

    if test_name in test_name_map:
        biomarker_name = test_name_map[test_name]
        biomarker_data = biomarker_regions.get(biomarker_name)
        
        if biomarker_data:
            # Get all possible color values including stable value
            color_values = []
            
            # Add stable value
            stable_data = biomarker_data.get('stable', {})
            if stable_data and 'rgb' in stable_data:
                color_values.append({
                    'value': stable_data.get('value', 'stable'),
                    'category': 'STABLE',
                    'rgb': stable_data['rgb']
                })
            
            # Add category values
            for category_name, category_data in biomarker_data['categories'].items():
                if 'rgb' in category_data:
                    rgb_value = category_data['rgb']
                    # Handle both single RGB values and lists of RGB values
                    if isinstance(rgb_value[0], list):
                        # Multiple RGB values
                        for single_rgb in rgb_value:
                            color_values.append({
                                'value': category_data.get('value', category_name),
                                'category': category_name,
                                'rgb': single_rgb
                            })
                    else:
                        # Single RGB value
                        color_values.append({
                            'value': category_data.get('value', category_name),
                            'category': category_name,
                            'rgb': rgb_value
                        })

            # Add individual values
            for value_data in biomarker_data['values']:
                color_values.append({
                    'value': value_data['value'],
                    'category': value_data['category'],
                    'rgb': value_data['rgb']
                })

            # Debug print for color matching
            print(f"\nColor matching for {test_name}:")
            print(f"Detected RGB: {rgb}")
            print("Available colors:")
            for cv in color_values:
                dist = color_distance(rgb, cv['rgb'])
                print(f"  {cv['category']}: RGB{cv['rgb']} - Distance: {dist:.2f}")

            # Find closest matching color
            closest_match = min(color_values, 
                              key=lambda x: color_distance(rgb, x['rgb']))
            
            print(f"Best match: {closest_match['category']} with RGB{closest_match['rgb']}")
            
            # Get category information
            category = closest_match['category']
            category_data = biomarker_data['categories'].get(category, {})
            
            result['value'] = closest_match['value']
            
            # Map clinical ranges to quantitative levels
            clinical_to_level = {
                'DEFICIENT': 'LOW',
                'NORMAL': 'MEDIUM',
                'EXCESS': 'HIGH',
                'SEVERE': 'VERY_HIGH',
                'OPTIMAL': 'MEDIUM',
                'ELEVATED': 'HIGH',
                'INCREASED': 'HIGH',
                'ADEQUATE': 'MEDIUM',
                'OVER_HYDRATION': 'LOW',
                'DEHYDRATION': 'HIGH',
                'SEVERE_DEHYDRATION': 'VERY_HIGH',
                'ACIDIC': 'LOW',
                'NEUTRAL': 'MEDIUM',
                'ALKALINE': 'HIGH',
                'NEGATIVE': 'LOW',
                'TRACE': 'LOW',
                'MODERATE': 'MEDIUM'
            }
            
            # Set level based on category
            for clinical_range in biomarker_data['clinical']:
                if clinical_range in category.upper():
                    result['level'] = clinical_to_level.get(clinical_range, 'MEDIUM')
                    break
            
            # Create description
            unit = biomarker_data['unit']
            value_str = f"{result['value']} {unit}" if unit else str(result['value'])
            category_label = category_data.get('label', category)
            result['description'] = f"{biomarker_name}: {value_str} - {category_label}"

    return result

def detect_dipstick_area(image):
    """Detect the main dipstick area and create a mask"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Améliorer la détection des bords noirs
    _, binary_mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Utiliser une méthode adaptative plus agressive
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,  # Augmenté pour plus de robustesse
        15   # Ajusté pour mieux détecter les bords
    )
    
    # Combiner les masques
    black_mask = cv2.bitwise_or(binary_mask, adaptive_mask)
    
    # Nettoyer le masque
    kernel = np.ones((3,3), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    
    # Trouver les contours
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours
    min_area = image.shape[0] * image.shape[1] * 0.05
    valid_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Approximer le contour pour le rendre plus régulier
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Obtenir le rectangle minimal
            rect = cv2.minAreaRect(approx)
            w = rect[1][0]
            h = rect[1][1]
            
            # S'assurer que la largeur est plus petite que la hauteur
            if w > h:
                w, h = h, w
                
            aspect_ratio = h / w
            if 1.5 < aspect_ratio < 8.0:
                valid_contours.append((approx, rect))
    
    if not valid_contours:
        return None, None, None
    
    # Prendre le plus grand contour valide
    main_contour, rect = max(valid_contours, key=lambda x: cv2.contourArea(x[0]))
    
    # Créer le masque pour la zone de la bandelette
    dipstick_mask = np.zeros_like(gray)
    cv2.drawContours(dipstick_mask, [main_contour], -1, (255), thickness=cv2.FILLED)
    
    # Dilater légèrement le masque pour inclure les bords
    kernel = np.ones((2,2), np.uint8)
    dipstick_mask = cv2.dilate(dipstick_mask, kernel, iterations=1)
    
    # Obtenir les coordonnées du rectangle englobant
    x, y, w, h = cv2.boundingRect(main_contour)
    
    return dipstick_mask, (x, y, w, h), image

def detect_black_rectangles(image):
    """Detect black rectangles in the image"""
    dipstick_mask, (x, y, w, h), _ = detect_dipstick_area(image)
    if dipstick_mask is None:
        return image, []
        
    working_image = image.copy()
    
    gray = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, black_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    black_mask = cv2.bitwise_and(black_mask, dipstick_mask)
    
    black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in black_contours:
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
        rect_x, rect_y = int(rect_x), int(rect_y)
        rect_w, rect_h = int(rect_w), int(rect_h)
        
        if (rect_x >= int(x) and rect_x + rect_w <= int(x + w) and 
            rect_y >= int(y) and rect_y + rect_h <= int(y + h)):
            cv2.rectangle(working_image, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), 
                         (255, 0, 0), 2)
            rectangles.append((rect_x, rect_y, rect_w, rect_h))

    return working_image, rectangles

def detect_colored_regions(image, start_index=1, min_area=300):
    """Detect colored regions in the image"""
    dipstick_mask, (x, y, w, h), image_with_white_bg = detect_dipstick_area(image)
    if dipstick_mask is None:
        return image, []
        
    working_image = image_with_white_bg.copy()
    
    # Convert coordinates to integers
    x, y, w, h = int(x), int(y), int(w), int(h)
    
    # Convert to HSV and split channels
    hsv = cv2.cvtColor(working_image, cv2.COLOR_BGR2HSV)
    h_channel, s, v = cv2.split(hsv)
    
    # Améliorer la saturation et la luminosité
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    s = cv2.add(s, 50)
    hsv = cv2.merge([h_channel, s, v])
    
    detected_regions = []
    current_index = start_index

    # Taille des rectangles
    min_rect_size = (w * h) * 0.008
    max_rect_size = (w * h) * 0.025

    filtered_colors = {k: v for k, v in COLOR_RANGES.items() if k in color_priority}
    sorted_colors = sorted(filtered_colors.items(), key=lambda x: color_priority.get(x[0], 999))

    kernel = np.ones((3,3), np.uint8)
    
    def check_overlap(rect1, rect2, threshold=0.3):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return False
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        min_area = min(area1, area2)
        
        return intersection / min_area > threshold

    for color_name, (lower, upper) in sorted_colors:
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.bitwise_and(mask, dipstick_mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_rect_size <= area <= max_rect_size:
                rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(contour)
                
                if (rect_x >= x and rect_x + rect_w <= x + w and 
                    rect_y >= y and rect_y + rect_h <= y + h):
                    
                    aspect_ratio = float(rect_w) / rect_h
                    if 0.7 < aspect_ratio < 1.3:
                        # Vérifier s'il y a chevauchement avec les régions existantes
                        is_duplicate = False
                        for existing in detected_regions:
                            if check_overlap(
                                (rect_x, rect_y, rect_w, rect_h),
                                existing['position']
                            ):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            roi = working_image[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]
                            avg_color = cv2.mean(roi)[:3]
                            
                            region = {
                                'position': (rect_x, rect_y, rect_w, rect_h),
                                'color': color_name,
                                'priority': color_priority.get(color_name, 999),
                                'rgb': (int(avg_color[2]), int(avg_color[1]), int(avg_color[0])),
                                'index': current_index
                            }
                            
                            detected_regions.append(region)
                            current_index += 1

    # Trier les régions par position
    detected_regions.sort(key=lambda x: (x['position'][1], x['position'][0]))

    # Assigner les noms de test
    mid_x = x + w//2
    left_regions = [r for r in detected_regions if r['position'][0] < mid_x]
    right_regions = [r for r in detected_regions if r['position'][0] >= mid_x]
    
    left_regions.sort(key=lambda r: r['position'][1])
    right_regions.sort(key=lambda r: r['position'][1])
    
    for i, region in enumerate(left_regions[:5]):
        region['test_name'] = static_positions['left_column'][i]
    
    for i, region in enumerate(right_regions[:5]):
        region['test_name'] = static_positions['right_column'][i]
    
    # Dessiner et interpréter
    all_regions = left_regions + right_regions
    
    for region in all_regions:
        rx, ry, rw, rh = region['position']
        test_name = region.get('test_name', 'Unknown')
        
        cv2.rectangle(working_image, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.putText(working_image, test_name, (rx, ry - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if test_name != 'Unknown':
            result = interpret_test_results(region)
            region['interpretation'] = result
            
            if result['level']:
                cv2.putText(working_image, result['level'],
                          (rx, ry + rh + 15),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return working_image, all_regions

def detect_qr_code(image):
    """
    Detect and decode QR code in the image (only black QR on white background)
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Only for black QR codes (no inversion)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            10
        )
        
        # Add preprocessing
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Save binary image for debugging
        cv2.imwrite('binary_debug.jpg', binary)
        
        # Detect QR codes
        qr_codes = decode(binary)
        
        if qr_codes:
            qr_code = qr_codes[0]
            qr_data = qr_code.data.decode('utf-8')
            
            points = qr_code.polygon
            if len(points) >= 4:
                points = np.array([[(p.x, p.y) for p in points]], dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)
                qr_coords = (x, y, w, h)
                
                debug_image = image.copy()
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                print(f"QR Code detected at coordinates: {qr_coords}")
                print(f"QR Code data: {qr_data}")
                
                return qr_data, qr_coords, debug_image
        
        return None, None, None

    except Exception as e:
        print(f"Error detecting QR code: {str(e)}")
        return None, None, None

def detect_qr_code_and_split(image):
    """Detect QR code and split the image at QR location"""
    dipstick_mask, (x, y, w, h), _ = detect_dipstick_area(image)
    working_image = image.copy()
    
    qr_data, qr_coords, debug_image = detect_qr_code(working_image)
    
    if qr_coords is not None:
        qr_x, qr_y, qr_w, qr_h = qr_coords
        if (qr_x >= x and qr_x + qr_w <= x + w and 
            qr_y >= y and qr_y + qr_h <= y + h):
            livv_y_position = qr_y - int((qr_y - y) * 0.15)
            squares_part = working_image[y:livv_y_position, x:x+w]
            livv_part = working_image[livv_y_position:qr_y, x:x+w]
            bottom_part = working_image[qr_y:y+h, x:x+w]
            return squares_part, livv_part, bottom_part, qr_y
    
    return image, None, None, None

def process_image(image_path):
    """Main function to process the image"""
    try:
        print(f"Attempting to read image from: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image from {image_path}")
            return None, None, None, None
        print(f"Successfully loaded image with shape: {image.shape}")

        # 1. D'abord détecter le QR et découper l'image
        squares_part, livv_part, bottom_part, qr_position = detect_qr_code_and_split(image.copy())
        
        if qr_position is None:
            print("No QR code detected")
            return None, None, None, None

        # 2. Détecter les couleurs AVANT les rectangles noirs
        squares_with_colors, colored_regions = detect_colored_regions(squares_part.copy())
        
        # 3. Ensuite détecter les rectangles noirs
        squares_with_rectangles, black_rectangles = detect_black_rectangles(squares_with_colors)
        
        print("\nSummary:")
        print("--------")
        print(f"Colored Regions detected: {len(colored_regions)}")  # Afficher d'abord les couleurs
        print(f"Black Rectangles detected: {len(black_rectangles)}")

        result_image = np.vstack((squares_with_rectangles, livv_part, bottom_part))
        return result_image, squares_with_rectangles, livv_part, bottom_part

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def main():
    """Main execution function"""
    try:
        image_path = "D:/livv testing/result.png"
        result_image, squares_part, livv_part, bottom_part = process_image(image_path)
        

        if result_image is not None and squares_part is not None:
            cv2.imwrite('output_full.jpg', result_image)
            cv2.imwrite('output_squares_part.jpg', squares_part)
            cv2.imwrite('output_livv_part.jpg', livv_part)
            
            # Create figure with subplot grid
            fig = plt.figure(figsize=(20, 8))
            
            # Image subplots (top row)
            plt.subplot(241)
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title('Full Image')
            plt.axis('off')
            
            plt.subplot(242)
            plt.imshow(cv2.cvtColor(squares_part, cv2.COLOR_BGR2RGB))
            plt.title('Squares Part')
            plt.axis('off')
            
            plt.subplot(243)
            plt.imshow(cv2.cvtColor(livv_part, cv2.COLOR_BGR2RGB))
            plt.title('LIVV Part')
            plt.axis('off')
            
            plt.subplot(244)
            plt.imshow(cv2.cvtColor(bottom_part, cv2.COLOR_BGR2RGB))
            plt.title('Bottom Part')
            plt.axis('off')
            
            # Results table (bottom row spanning all columns)
            plt.subplot(212)
            plt.axis('off')
            
            # Create table data
            table_data = []
            headers = ['Biomarker', 'Value', 'Level', 'Description']
            
            # Get results from colored regions
            _, colored_regions = detect_colored_regions(squares_part)
            
            for region in colored_regions:
                if 'test_name' in region and 'interpretation' in region:
                    result = region['interpretation']
                    table_data.append([
                        region['test_name'],
                        result.get('value', 'N/A'),
                        result.get('level', 'N/A'),
                        result.get('description', 'N/A')
                    ])
            
            # Create table only if we have data
            if table_data:
                table = plt.table(
                    cellText=table_data,
                    colLabels=headers,
                    loc='center',
                    cellLoc='center',
                    colColours=['#f2f2f2']*4,
                    cellColours=[['#ffffff']*4 for _ in range(len(table_data))],
                    bbox=[0.1, 0.0, 0.8, 0.9]
                )
                
                # Style the table
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                
                plt.title('Biomarker Results', pad=20)
            else:
                plt.text(0.5, 0.5, 'No biomarkers detected', 
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=12)
                plt.title('No Results Available', pad=20)
            
            plt.tight_layout()
            plt.show()
            
            print("\nProcessing completed successfully!")
        else:
            print("Image processing failed - one or more parts are None")
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()