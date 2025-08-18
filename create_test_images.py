#!/usr/bin/env python3
"""
Create test images for the crop disease detection app.
This script generates sample leaf images that you can use to test your app.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random

def create_test_image(filename, title, color_scheme, disease_pattern=None):
    """Create a synthetic leaf image for testing."""
    
    # Create a 192x192 image (matching your model's input size)
    width, height = 192, 192
    
    # Create base leaf shape
    img = Image.new('RGB', (width, height), color=(50, 80, 50))  # Dark green background
    draw = ImageDraw.Draw(img)
    
    # Draw leaf shape
    leaf_color = color_scheme['base']
    
    # Main leaf body (ellipse)
    margin = 20
    draw.ellipse([margin, margin, width-margin, height-margin], fill=leaf_color)
    
    # Add leaf veins
    vein_color = tuple(max(0, c-30) for c in leaf_color)  # Darker version of base color
    
    # Central vein
    draw.line([(width//2, margin), (width//2, height-margin)], fill=vein_color, width=3)
    
    # Side veins
    for i in range(3):
        y_pos = margin + (height-2*margin) * (i+1) // 4
        draw.line([(width//2, y_pos), (margin+10, y_pos-10)], fill=vein_color, width=2)
        draw.line([(width//2, y_pos), (width-margin-10, y_pos-10)], fill=vein_color, width=2)
    
    # Add disease patterns if specified
    if disease_pattern == 'spots':
        # Add brown/dark spots for diseases like bacterial spot
        spot_color = (101, 67, 33)  # Brown color
        for _ in range(random.randint(3, 8)):
            x = random.randint(margin+10, width-margin-20)
            y = random.randint(margin+10, height-margin-20)
            radius = random.randint(5, 15)
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=spot_color)
    
    elif disease_pattern == 'yellowing':
        # Add yellow patches for diseases like early blight
        yellow_overlay = Image.new('RGBA', (width, height), (255, 255, 0, 50))
        img = Image.alpha_composite(img.convert('RGBA'), yellow_overlay).convert('RGB')
    
    elif disease_pattern == 'browning':
        # Add brown edges for diseases like late blight
        brown_color = (139, 69, 19)
        for i in range(10):
            x = random.randint(0, width)
            y = random.randint(0, height)
            if x < margin+20 or x > width-margin-20 or y < margin+20 or y > height-margin-20:
                draw.ellipse([x-8, y-8, x+8, y+8], fill=brown_color)
    
    # Add some texture/noise to make it more realistic
    pixels = np.array(img)
    noise = np.random.randint(-20, 20, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    # Add title at the bottom
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add a small label
    draw = ImageDraw.Draw(img)
    text_bbox = draw.textbbox((0, 0), title, font=font) if font else (0, 0, len(title)*6, 10)
    text_width = text_bbox[2] - text_bbox[0]
    text_x = (width - text_width) // 2
    draw.text((text_x, height-20), title, fill=(255, 255, 255), font=font)
    
    return img

def create_test_images():
    """Create a set of test images for the crop disease detection app."""
    
    # Create test_images directory
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    print("🖼️ Creating test images for your crop disease detection app...")
    
    # Define test cases based on your model's classes
    test_cases = [
        # Healthy samples
        {
            'filename': 'tomato_healthy.jpg',
            'title': 'Healthy Tomato',
            'color_scheme': {'base': (34, 139, 34)},  # Forest green
            'disease_pattern': None
        },
        {
            'filename': 'potato_healthy.jpg', 
            'title': 'Healthy Potato',
            'color_scheme': {'base': (50, 205, 50)},  # Lime green
            'disease_pattern': None
        },
        {
            'filename': 'pepper_healthy.jpg',
            'title': 'Healthy Pepper', 
            'color_scheme': {'base': (0, 128, 0)},  # Green
            'disease_pattern': None
        },
        
        # Diseased samples
        {
            'filename': 'tomato_bacterial_spot.jpg',
            'title': 'Tomato Bacterial Spot',
            'color_scheme': {'base': (107, 142, 35)},  # Olive green
            'disease_pattern': 'spots'
        },
        {
            'filename': 'tomato_early_blight.jpg',
            'title': 'Tomato Early Blight',
            'color_scheme': {'base': (154, 205, 50)},  # Yellow green
            'disease_pattern': 'yellowing'
        },
        {
            'filename': 'potato_late_blight.jpg',
            'title': 'Potato Late Blight',
            'color_scheme': {'base': (85, 107, 47)},  # Dark olive green
            'disease_pattern': 'browning'
        },
        {
            'filename': 'pepper_bacterial_spot.jpg',
            'title': 'Pepper Bacterial Spot',
            'color_scheme': {'base': (128, 128, 0)},  # Olive
            'disease_pattern': 'spots'
        },
        
        # Additional test cases
        {
            'filename': 'tomato_leaf_mold.jpg',
            'title': 'Tomato Leaf Mold',
            'color_scheme': {'base': (143, 188, 143)},  # Dark sea green
            'disease_pattern': 'spots'
        }
    ]
    
    # Create each test image
    for i, test_case in enumerate(test_cases, 1):
        print(f"  Creating {i}/{len(test_cases)}: {test_case['title']}")
        
        img = create_test_image(
            test_case['filename'],
            test_case['title'],
            test_case['color_scheme'],
            test_case.get('disease_pattern')
        )
        
        # Save the image
        img_path = os.path.join(test_dir, test_case['filename'])
        img.save(img_path, 'JPEG', quality=85)
    
    print(f"\n✅ Created {len(test_cases)} test images in the '{test_dir}' directory!")
    print("\n📋 Test images created:")
    for test_case in test_cases:
        print(f"  • {test_case['filename']} - {test_case['title']}")
    
    print(f"\n🧪 How to use these images:")
    print(f"  1. Go to your Streamlit app")
    print(f"  2. Click 'Upload Image' tab")
    print(f"  3. Upload any of the images from the '{test_dir}' folder")
    print(f"  4. See how your model predicts different diseases!")
    
    print(f"\n💡 Note: These are synthetic images for testing the app interface.")
    print(f"    For real disease detection, use actual photos of plant leaves.")

if __name__ == "__main__":
    create_test_images()
