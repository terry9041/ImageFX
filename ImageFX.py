import tkinter as tk
from tkinter import filedialog, Menu,  Toplevel, Scale, HORIZONTAL
import struct
import numpy as np
from scipy.signal import convolve2d
import random


class BmpImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("BMP Image Viewer")
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Set initial window size to be wider
        self.root.geometry("800x600")

        # Create menu bar
        self.menu_bar = Menu(root)
        root.config(menu=self.menu_bar)

        # Core Operations menu
        core_menu = Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Core Operations", menu=core_menu)
        core_menu.add_command(label="Open File", command=self.open_file)
        core_menu.add_command(label="Exit", command=root.quit)
        core_menu.add_command(label="Grayscale", command=self.grayscale)
        core_menu.add_command(label="Ordered Dithering", command=self.ordered_dithering)
        core_menu.add_command(label="Auto Level", command=self.auto_level)

        # Optional Operations menu (to be added after opening an image)
        self.optional_menu = Menu(self.menu_bar, tearoff=0)
        self.optional_menu.add_command(label="Invert Colors", command=self.invert_colors)
        self.optional_menu.add_command(label="Sepia Tone", command=self.sepia_tone)
        self.optional_menu.add_command(label="Fish Eye Effect", command=self.fish_eye_effect)
        self.optional_menu.add_command(label="Mosaic Effect", command=self.mosaic_effect)
        self.optional_menu.add_command(label="Sobel Edge Detection", command=self.sobel_edge_detection)
        self.optional_menu.add_command(label="Canny Edge Detection", command=self.canny_edge_detection)
        self.optional_menu.add_command(label="Blur Image", command=self.blur_image)
        self.optional_menu.add_command(label="Histogram Equalization", command=self.histogram_equalization)
        self.optional_menu.add_command(label="Unsharp Masking", command=self.unsharp_masking)
        self.optional_menu.add_command(label="Glitch Effect", command=self.glitch_effect)
        # self.optional_menu.add_command(label="laplacian_sharpen", command=self.laplacian_sharpen)
        self.optional_menu_added = False
        
        
    def close_slider_window(self):
        """Close the slider window if it exists."""
        if hasattr(self, 'slider_window') and self.slider_window is not None:
            self.slider_window.destroy()
            self.slider_window = None

    def glitch_effect(self):
        self.close_slider_window()


        # Create a slider window to adjust glitch intensity
        self.slider_window = tk.Toplevel(self.root)
        self.slider_window.title("Glitch Effect - Adjust Intensity")
        intensity_scale = tk.Scale(self.slider_window, from_=0, to=50, orient=tk.HORIZONTAL,
                                label="Glitch Intensity", length=300)
        intensity_scale.set(10)
        intensity_scale.pack()

        def apply_glitch(intensity):
            # Convert the original pixel data to a NumPy array
            pixel_data = np.array(self.original_pixel_data)
            height, width = pixel_data.shape[:2]
            glitched = np.copy(pixel_data)

            # For each row, shift the pixels by a random offset within [-intensity, intensity]
            for y in range(height):
                offset = random.randint(-intensity, intensity)
                # np.roll shifts pixels along the width dimension (axis 0 of the row slice)
                glitched[y, :] = np.roll(pixel_data[y, :], shift=offset, axis=0)

            # Update the display with the glitched image
            self.display_side_by_side(self.original_pixel_data, glitched.tolist(), width, height)

        # Callback that applies the glitch effect on slider update
        def on_slider_change(val):
            apply_glitch(int(intensity_scale.get()))

        intensity_scale.config(command=on_slider_change)
        # Apply initial glitch effect
        apply_glitch(intensity_scale.get())
    

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            self.display_image(file_path)
            # Add the optional menu after opening an image
            if not self.optional_menu_added:
                self.menu_bar.add_cascade(label="Optional Operations", menu=self.optional_menu)
                self.optional_menu_added = True

    def display_image(self, file_path):
        with open(file_path, 'rb') as bmp_file:
            header = bmp_file.read(14)
            file_type, file_size, reserved_1, reserved_2, pixel_offset = struct.unpack('<2sIHHI', header)
            dib_header = bmp_file.read(40)
            (header_size, width, height, planes, bits_per_pixel,
             compression, image_size, h_res, v_res,
             colors_in_palette, important_colors) = struct.unpack('<IIIHHIIIIII', dib_header)
            if bits_per_pixel != 24 or compression != 0:
                print("Not a 24-bit uncompressed BMP file.")
                return
            row_size = (width * 3 + 3) & ~3
            padding = row_size - width * 3
            pixel_data = []
            bmp_file.seek(pixel_offset)
            for y in range(height):
                row = []
                for x in range(width):
                    bgr = bmp_file.read(3)
                    row.append((bgr[2], bgr[1], bgr[0]))
                bmp_file.read(padding)
                pixel_data.append(row)
            pixel_data.reverse()
            self.original_pixel_data = pixel_data
            self.width = width
            self.height = height
            self.root.geometry(f"{width}x{height}")
            self.display_tk_image(pixel_data, width, height)
            self.close_slider_window()

    def display_tk_image(self, pixel_data, width, height):
        img = tk.PhotoImage(width=width, height=height)
        for y in range(height):
            for x in range(width):
                hex_color = '#{:02x}{:02x}{:02x}'.format(*pixel_data[y][x])
                img.put(hex_color, (x, y))
        self.image_label.config(image=img)
        self.image_label.image = img

    def display_side_by_side(self, left_image_data, right_image_data, width, height):
        total_width = width * 2
        img = tk.PhotoImage(width=total_width, height=height)
        for y in range(height):
            for x in range(width):
                hex_color = '#{:02x}{:02x}{:02x}'.format(*left_image_data[y][x])
                img.put(hex_color, (x, y))
        for y in range(height):
            for x in range(width):
                hex_color = '#{:02x}{:02x}{:02x}'.format(*right_image_data[y][x])
                img.put(hex_color, (x + width, y))
        self.root.geometry(f"{total_width}x{height}")
        self.image_label.config(image=img)
        self.image_label.image = img

    def emboss_effect(self):
        self.close_slider_window()
        import numpy as np
        from scipy.signal import convolve2d

        emboss_kernel = np.array([[-2, -1, 0],
                                [-1,  1, 1],
                                [ 0,  1, 2]])
        # Separate color channels
        r_channel = np.array([[pixel[0] for pixel in row] for row in self.original_pixel_data])
        g_channel = np.array([[pixel[1] for pixel in row] for row in self.original_pixel_data])
        b_channel = np.array([[pixel[2] for pixel in row] for row in self.original_pixel_data])

        # Apply convolution to each channel
        r_embossed = convolve2d(r_channel, emboss_kernel, mode='same', boundary='symm')
        g_embossed = convolve2d(g_channel, emboss_kernel, mode='same', boundary='symm')
        b_embossed = convolve2d(b_channel, emboss_kernel, mode='same', boundary='symm')

        embossed_pixel_data = []
        for y in range(self.height):
            new_row = []
            for x in range(self.width):
                # Add 128 to recenter the intensities and clip to [0,255]
                r = np.clip(r_embossed[y, x] + 128, 0, 255)
                g = np.clip(g_embossed[y, x] + 128, 0, 255)
                b = np.clip(b_embossed[y, x] + 128, 0, 255)
                new_row.append((int(r), int(g), int(b)))
            embossed_pixel_data.append(new_row)
        
        self.display_side_by_side(self.original_pixel_data, embossed_pixel_data, self.width, self.height)

    def grayscale(self):
        self.close_slider_window()
        gray_pixel_data = []
        for row in self.original_pixel_data:
            gray_row = []
            for pixel in row:
                gray_value = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
                gray_row.append((gray_value, gray_value, gray_value))
            gray_pixel_data.append(gray_row)
        self.display_side_by_side(self.original_pixel_data, gray_pixel_data, self.width, self.height)

    def ordered_dithering(self):
        self.close_slider_window()
        gray_pixel_data = []
        for row in self.original_pixel_data:
            gray_row = []
            for pixel in row:
                gray_value = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
                gray_row.append(gray_value)
            gray_pixel_data.append(gray_row)
        bayer_matrix =  [
    [  0, 136,  34, 170],
    [204,  68, 238, 102],
    [ 51, 187,  17, 153],
    [255, 119, 221,  85]
]
        bayer_size = len(bayer_matrix)
        dithered_pixel_data = []
        for y in range(self.height):
            dithered_row = []
            for x in range(self.width):
                threshold = bayer_matrix[y % bayer_size][x % bayer_size]
                gray_value = gray_pixel_data[y][x]
                dithered_value = 255 if gray_value > threshold else 0
                dithered_row.append((dithered_value, dithered_value, dithered_value))
            dithered_pixel_data.append(dithered_row)
        self.display_side_by_side(self.original_pixel_data, dithered_pixel_data, self.width, self.height)

    def auto_level(self):
        self.close_slider_window()
        flat_pixels = [pixel for row in self.original_pixel_data for pixel in row]
        reds = [pixel[0] for pixel in flat_pixels]
        greens = [pixel[1] for pixel in flat_pixels]
        blues = [pixel[2] for pixel in flat_pixels]
        min_r, max_r = min(reds), max(reds)
        min_g, max_g = min(greens), max(greens)
        min_b, max_b = min(blues), max(blues)
        auto_leveled_pixel_data = []
        for row in self.original_pixel_data:
            new_row = []
            for pixel in row:
                r = int((pixel[0] - min_r) * 255 / (max_r - min_r)) if max_r > min_r else pixel[0]
                g = int((pixel[1] - min_g) * 255 / (max_g - min_g)) if max_g > min_g else pixel[1]
                b = int((pixel[2] - min_b) * 255 / (max_b - min_b)) if max_b > min_b else pixel[2]
                new_row.append((r, g, b))
            auto_leveled_pixel_data.append(new_row)
        self.display_side_by_side(self.original_pixel_data, auto_leveled_pixel_data, self.width, self.height)

    # Optional Operations
    def invert_colors(self):
        self.close_slider_window()
        inverted_pixel_data = []
        for row in self.original_pixel_data:
            new_row = []
            for pixel in row:
                r = 255 - pixel[0]
                g = 255 - pixel[1]
                b = 255 - pixel[2]
                new_row.append((r, g, b))
            inverted_pixel_data.append(new_row)
        self.display_side_by_side(self.original_pixel_data, inverted_pixel_data, self.width, self.height)

    def sepia_tone(self):
        self.close_slider_window()
        sepia_pixel_data = []
        for row in self.original_pixel_data:
            sepia_row = []
            for pixel in row:
                r, g, b = pixel
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                tr = min(255, tr)
                tg = min(255, tg)
                tb = min(255, tb)
                sepia_row.append((tr, tg, tb))
            sepia_pixel_data.append(sepia_row)
        self.display_side_by_side(self.original_pixel_data, sepia_pixel_data, self.width, self.height)

    def fish_eye_effect(self):
        self.close_slider_window()
        pixel_data = np.array(self.original_pixel_data, dtype=np.uint8)
        height, width = pixel_data.shape[:2]
        dst_pixels = np.zeros_like(pixel_data)

        for y in range(height):
            ny = (2 * y) / height - 1
            ny2 = ny * ny
            for x in range(width):
                nx = (2 * x) / width - 1
                nx2 = nx * nx
                r = np.sqrt(nx2 + ny2)
                if 0.0 <= r <= 1.0:
                    nr = (r + (1 - np.sqrt(1 - r * r))) / 2
                    if nr <= 1.0:
                        theta = np.arctan2(ny, nx)
                        nxn = nr * np.cos(theta)
                        nyn = nr * np.sin(theta)
                        x2 = int(((nxn + 1) * width) / 2)
                        y2 = int(((nyn + 1) * height) / 2)
                        if 0 <= x2 < width and 0 <= y2 < height:
                            dst_pixels[y, x] = pixel_data[y2, x2]
        new_pixel_data = dst_pixels.tolist()
        self.display_side_by_side(self.original_pixel_data, new_pixel_data, width, height)

    def mosaic_effect(self, block_size=10):
        """
        Apply mosaic effect to the image with an interactive slider for block size.

        Parameters:
        block_size (int): Size of the mosaic blocks.
        """
        # Close existing slider window if it exists
        self.close_slider_window()

        def apply_changes():
            block_size = block_size_slider.get()
            update_image(block_size)

        def update_image(block_size):
            height = self.height
            width = self.width
            mosaic_pixel_data = []

            for y in range(0, height, block_size):
                for dy in range(block_size):
                    if y + dy >= height:
                        continue
                    row = []
                    for x in range(0, width, block_size):
                        block_pixels = []
                        for dx in range(block_size):
                            if x + dx < width:
                                pixel = self.original_pixel_data[y + dy][x + dx]
                                block_pixels.append(pixel)
                        if block_pixels:
                            avg_r = sum(pixel[0] for pixel in block_pixels) // len(block_pixels)
                            avg_g = sum(pixel[1] for pixel in block_pixels) // len(block_pixels)
                            avg_b = sum(pixel[2] for pixel in block_pixels) // len(block_pixels)
                            avg_color = (avg_r, avg_g, avg_b)
                            row.extend([avg_color] * len(block_pixels))
                    mosaic_pixel_data.append(row)

            self.display_side_by_side(self.original_pixel_data, mosaic_pixel_data, self.width, self.height)
        
        # Initial image update
        update_image(block_size)

        # Create a new window for sliders
        self.slider_window = Toplevel(self.root)
        self.slider_window.title("Adjust Mosaic Effect Parameters")

        # Force update of the main window's geometry
        self.root.update_idletasks()

        
        # Position the slider window so it doesn't cover the main window
        # Get the geometry of the main window
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()

        # Set the geometry of the slider window to appear next to the main window
        self.slider_window.geometry(f"+{x + root_width + 10}+{y}")

        # Block Size Slider
        block_size_slider = Scale(self.slider_window, from_=1, to=50, resolution=1, orient=HORIZONTAL, label="Block Size")
        block_size_slider.set(block_size)
        block_size_slider.pack(fill='x', expand=True, padx=10, pady=5)

        # Apply Button
        apply_button = tk.Button(self.slider_window, text="Apply", command=apply_changes)
        apply_button.pack()

        

    def sobel_edge_detection(self):
        self.close_slider_window()
        edge_pixel_data = self.original_pixel_data
        # Step 1: Grayscale Conversion
        gray_pixel_data = []
        for row in edge_pixel_data:
            gray_row = []
            for pixel in row:
                gray_value = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
                gray_row.append(gray_value)
            gray_pixel_data.append(gray_row)
        
        gray_array = np.array(gray_pixel_data)
        
        # Step 2: Gradient Calculation (Sobel Operators)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        gradient_x = convolve2d(gray_array, sobel_x, mode='same', boundary='symm')
        gradient_y = convolve2d(gray_array, sobel_y, mode='same', boundary='symm')
        
        gradient_magnitude = np.hypot(gradient_x, gradient_y)
        
        # Normalize the gradient magnitude to range [0, 255]
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255
        gradient_magnitude = gradient_magnitude.astype(np.uint8)
        
        # Convert to RGB format for consistency
        edge_pixel_data = [[(v, v, v) for v in row] for row in gradient_magnitude]
        
        self.display_side_by_side(self.original_pixel_data, edge_pixel_data, self.width, self.height)


    def canny_edge_detection(self, low_threshold=50, high_threshold=150):
        edge_pixel_data = self.original_pixel_data
        # Step 1: Grayscale Conversion
        gray_pixel_data = []
        for row in edge_pixel_data:
            gray_row = []
            for pixel in row:
                gray_value = int(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])
                gray_row.append(gray_value)
            gray_pixel_data.append(gray_row)
        
        gray_array = np.array(gray_pixel_data)
        
        # Step 2: Gaussian Blur
        gaussian_kernel = np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]]) / 16
        blurred_image = convolve2d(gray_array, gaussian_kernel, mode='same', boundary='symm')
        
        # Step 3: Gradient Calculation (Sobel Operators)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        gradient_x = convolve2d(blurred_image, sobel_x, mode='same', boundary='symm')
        gradient_y = convolve2d(blurred_image, sobel_y, mode='same', boundary='symm')
        
        gradient_magnitude = np.hypot(gradient_x, gradient_y)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * 180.0 / np.pi
        gradient_direction[gradient_direction < 0] += 180  # Normalize angles to 0-180
        
        # Step 4: Non-Maximum Suppression
        M, N = gradient_magnitude.shape
        non_max_suppressed = np.zeros((M, N), dtype=np.int32)
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    angle = gradient_direction[i, j]
                    magnitude = gradient_magnitude[i, j]
                    
                    # Determine the neighbors based on direction
                    if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                        q, r = gradient_magnitude[i, j + 1], gradient_magnitude[i, j - 1]
                    elif (22.5 <= angle < 67.5):
                        q, r = gradient_magnitude[i + 1, j - 1], gradient_magnitude[i - 1, j + 1]
                    elif (67.5 <= angle < 112.5):
                        q, r = gradient_magnitude[i + 1, j], gradient_magnitude[i - 1, j]
                    elif (112.5 <= angle < 157.5):
                        q, r = gradient_magnitude[i - 1, j - 1], gradient_magnitude[i + 1, j + 1]
                    
                    if magnitude >= q and magnitude >= r:
                        non_max_suppressed[i, j] = magnitude
                    else:
                        non_max_suppressed[i, j] = 0
                except IndexError:
                    pass
        
        # Step 5: Double Thresholding
        strong_pixel = 255
        weak_pixel = 25
        
        strong_edges = (non_max_suppressed >= high_threshold)
        weak_edges = ((non_max_suppressed >= low_threshold) & (non_max_suppressed < high_threshold))
        
        thresholded_image = np.zeros_like(non_max_suppressed, dtype=np.uint8)
        thresholded_image[strong_edges] = strong_pixel
        thresholded_image[weak_edges] = weak_pixel
        
        # Step 6: Edge Tracking by Hysteresis
        final_edges = np.copy(thresholded_image)
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if final_edges[i, j] == weak_pixel:
                    if ((final_edges[i+1, j-1] == strong_pixel) or (final_edges[i+1, j] == strong_pixel) or 
                        (final_edges[i+1, j+1] == strong_pixel) or (final_edges[i, j-1] == strong_pixel) or 
                        (final_edges[i, j+1] == strong_pixel) or (final_edges[i-1, j-1] == strong_pixel) or 
                        (final_edges[i-1, j] == strong_pixel) or (final_edges[i-1, j+1] == strong_pixel)):
                        final_edges[i, j] = strong_pixel
                    else:
                        final_edges[i, j] = 0
        
        # Convert to RGB format for consistency
        edge_pixel_data = [[(v, v, v) for v in row] for row in final_edges]
        
        self.display_side_by_side(self.original_pixel_data, edge_pixel_data, self.width, self.height)


    def blur_image(self, kernel_size=5):
        """
        Apply blur effect to the image with an interactive slider for kernel size.

        Parameters:
        kernel_size (int): Size of the blur kernel.
        """
        # Close existing slider window if it exists
        self.close_slider_window()

        def apply_changes():
            kernel_size = kernel_size_slider.get()
            update_image(kernel_size)

        def update_image(kernel_size):
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            r_channel = np.array([[pixel[0] for pixel in row] for row in self.original_pixel_data])
            g_channel = np.array([[pixel[1] for pixel in row] for row in self.original_pixel_data])
            b_channel = np.array([[pixel[2] for pixel in row] for row in self.original_pixel_data])
            r_blurred = convolve2d(r_channel, kernel, mode='same', boundary='symm')
            g_blurred = convolve2d(g_channel, kernel, mode='same', boundary='symm')
            b_blurred = convolve2d(b_channel, kernel, mode='same', boundary='symm')
            blurred_pixel_data = []
            for y in range(self.height):
                new_row = []
                for x in range(self.width):
                    r = int(r_blurred[y, x])
                    g = int(g_blurred[y, x])
                    b = int(b_blurred[y, x])
                    new_row.append((r, g, b))
                blurred_pixel_data.append(new_row)
            self.display_side_by_side(self.original_pixel_data, blurred_pixel_data, self.width, self.height)

        
        # Initial image update
        update_image(kernel_size)

        # Create a new window for sliders
        self.slider_window = Toplevel(self.root)
        self.slider_window.title("Adjust Blur Parameters")

        # Force update of the main window's geometry
        self.root.update_idletasks()


        # Position the slider window so it doesn't cover the main window
        # Get the geometry of the main window
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()

        # Set the geometry of the slider window to appear next to the main window
        self.slider_window.geometry(f"+{x + root_width + 10}+{y}")

        # Kernel Size Slider
        kernel_size_slider = Scale(self.slider_window, from_=1, to=50, resolution=1, orient=HORIZONTAL, label="Blur Level")
        kernel_size_slider.set(kernel_size)
        kernel_size_slider.pack(fill='x', expand=True, padx=10, pady=5)
        
        # Apply Button
        apply_button = tk.Button(self.slider_window, text="Apply", command=apply_changes)
        apply_button.pack()

    def histogram_equalization(self):
        self.close_slider_window()
        # Convert the image to a NumPy array
        pixel_data = np.array(self.original_pixel_data, dtype=np.uint8)

        # Convert RGB to YCbCr color space
        ycbcr_image = []
        for row in pixel_data:
            ycbcr_row = []
            for pixel in row:
                r, g, b = pixel
                y = 0.299 * r + 0.587 * g + 0.114 * b
                cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
                cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
                ycbcr_row.append([y, cb, cr])
            ycbcr_image.append(ycbcr_row)
        ycbcr_image = np.array(ycbcr_image, dtype=np.float32)

        # Extract the Y channel
        y_channel = ycbcr_image[:, :, 0]

        # Flatten the Y channel and compute histogram
        flat_y = y_channel.flatten()
        histogram, bins = np.histogram(flat_y, bins=256, range=[0, 255])

        # Compute the cumulative distribution function (CDF)
        cdf = np.cumsum(histogram)
        cdf_normalized = cdf * 255 / cdf[-1]

        # Use linear interpolation of the CDF to find new pixel values
        equalized_y = np.interp(flat_y, bins[:-1], cdf_normalized)
        equalized_y = equalized_y.reshape(y_channel.shape)

        # Update the Y channel with equalized values
        ycbcr_image[:, :, 0] = equalized_y

        # Convert back to RGB color space
        equalized_pixel_data = []
        for row in ycbcr_image:
            rgb_row = []
            for y, cb, cr in row:
                y -= 16
                cb -= 128
                cr -= 128
                r = 1.164 * y + 1.596 * cr
                g = 1.164 * y - 0.392 * cb - 0.813 * cr
                b = 1.164 * y + 2.017 * cb
                r = np.clip(r, 0, 255)
                g = np.clip(g, 0, 255)
                b = np.clip(b, 0, 255)
                rgb_row.append((int(r), int(g), int(b)))
            equalized_pixel_data.append(rgb_row)

        self.display_side_by_side(self.original_pixel_data, equalized_pixel_data, self.width, self.height)
        

    def unsharp_masking(self, amount=1.5, radius=1.0, threshold=10):
        """
        Apply unsharp masking to the image with interactive sliders.

        Parameters:
        amount (float): Strength of the sharpening effect.
        radius (float): Radius for Gaussian blur.
        threshold (int): Minimum brightness change to be sharpened.
        """
        # Close existing slider window if it exists
        self.close_slider_window()

        def apply_changes():
            amount = amount_slider.get()
            radius = radius_slider.get()
            threshold = threshold_slider.get()
            update_image(amount, radius, threshold)

        def update_image(amount, radius, threshold):
            height = len(self.original_pixel_data)
            width = len(self.original_pixel_data[0])
            
            # Convert original image to numpy array
            image_array = np.array(self.original_pixel_data, dtype=np.float32)
            
            # Create Gaussian kernel
            def gaussian_kernel(size, sigma):
                """Generates a 2D Gaussian kernel."""
                ax = np.arange(-size // 2 + 1., size // 2 + 1.)
                xx, yy = np.meshgrid(ax, ax)
                kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
                return kernel / np.sum(kernel)
            
            # Determine the kernel size (commonly 6*sigma rounded up to next odd integer)
            kernel_size = int(6 * radius) | 1  # Ensure it's odd
            kernel = gaussian_kernel(kernel_size, radius)
            
            # Initialize output image
            sharpened_image = np.copy(image_array)
            
            for channel in range(3):
                # Apply manual Gaussian blur to the channel using convolve2d
                blurred = convolve2d(image_array[:, :, channel], kernel, mode='same', boundary='symm')
                # Create the mask
                mask = image_array[:, :, channel] - blurred
                # Apply threshold
                mask[np.abs(mask) < threshold] = 0
                # Enhance the image by adding the mask scaled by amount
                sharpened = image_array[:, :, channel] + amount * mask
                # Clip values to [0, 255]
                sharpened = np.clip(sharpened, 0, 255)
                sharpened_image[:, :, channel] = sharpened
            
            # Convert to uint8
            sharpened_image = sharpened_image.astype(np.uint8)
            sharpened_pixel_data = sharpened_image.tolist()
            
            self.display_side_by_side(self.original_pixel_data, sharpened_pixel_data, self.width, self.height)

        # Initial image update
        update_image(amount, radius, threshold)

        # Create a new window for sliders
        self.slider_window = Toplevel(self.root)
        self.slider_window.title("Adjust Unsharp Masking Parameters")

        # Force update of the main window's geometry
        self.root.update_idletasks()

        # Position the slider window so it doesn't cover the main window
        # Get the geometry of the main window
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()

        # Set the geometry of the slider window to appear next to the main window
        self.slider_window.geometry(f"+{x + root_width + 10}+{y}")

        # Amount Slider
        amount_slider = Scale(self.slider_window, from_=0.1, to=5.0, resolution=0.1,
                            orient=HORIZONTAL, label="Amount")
        amount_slider.set(amount)
        amount_slider.pack(fill='x', expand=True, padx=10, pady=5)

        # Radius Slider
        radius_slider = Scale(self.slider_window, from_=0.1, to=10.0, resolution=0.1,
                            orient=HORIZONTAL, label="Radius")
        radius_slider.set(radius)
        radius_slider.pack(fill='x', expand=True, padx=10, pady=5)

        # Threshold Slider
        threshold_slider = Scale(self.slider_window, from_=0, to=255, resolution=1,
                                orient=HORIZONTAL, label="Threshold")
        threshold_slider.set(threshold)
        threshold_slider.pack(fill='x', expand=True, padx=10, pady=5)

        # Apply Button
        apply_button = tk.Button(self.slider_window, text="Apply", command=apply_changes)
        apply_button.pack(pady=10)
if __name__ == "__main__":
    root = tk.Tk()
    app = BmpImageViewer(root)
    root.mainloop()