#%%
from scipy.ndimage import convolve
import tkinter as tk
from PIL import Image
from PIL import ImageDraw, ImageTk
from tkinter import ttk
from PIL import ImageTk
from tkinter import filedialog
from ttkthemes import ThemedStyle
import cv2
import numpy as np
import tkinter.filedialog
import tkinter
import tkinter.messagebox
import customtkinter


customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

once = True
img_screenshot = None

class App(customtkinter.CTk):
    original_image = None
    changed_image = None
    edited_image = None
    def __init__(self):
        super().__init__()

        # configure window
        self.title("FinalProject.py")
        self.geometry(f"{1280}x{720}") 

        self.logo_image = customtkinter.CTkImage(Image.open("image/iconshow.png"))
        self.iconbitmap("image/iconshow.ico")
        

        # Đọc hình ảnh từ file
        image = Image.open("image/save1.png")
        export_img=Image.open("image/export.png")
        draw_img=Image.open("image/draw.png")
        open_img=Image.open("image/open.png")
        reset_img=Image.open("image/reset.png")
        avatar_img=Image.open("image/iconshow1.png")
        
        # Chuyển đổi hình ảnh thành đối tượng PhotoImage
        
        photo1 = ImageTk.PhotoImage(image)
        export=ImageTk.PhotoImage(export_img)
        open=ImageTk.PhotoImage(open_img)
        reset=ImageTk.PhotoImage(reset_img)
        draw=ImageTk.PhotoImage(draw_img)
        avatar=ImageTk.PhotoImage(avatar_img)

        self.info_frame=customtkinter.CTkFrame(self,fg_color="transparent")

        self.info_frame.grid(row=3,column=3,padx=20,pady=(10,10))

        # Banner
        self.member = customtkinter.CTkLabel(self.info_frame, text="Member:",font=customtkinter.CTkFont(size=16, weight="bold"), compound="left",anchor="w")
        self.member.grid(row=0, column=0, sticky="nsew")
        self.info = customtkinter.CTkLabel(self.info_frame, text="Le Hoang Lam",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info.grid(row=1, column=1, sticky="nsew")
        self.info1 = customtkinter.CTkLabel(self.info_frame, text="Nguyen Hoang Nhan",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info1.grid(row=2, column=1, sticky="nsew")
        self.info2 = customtkinter.CTkLabel(self.info_frame, text="Nguyen Viet Anh",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info2.grid(row=3, column=1, sticky="nsew")
        self.info3 = customtkinter.CTkLabel(self.info_frame, text="Le Y Thien",font=customtkinter.CTkFont(size=14, weight="normal"),compound="left",anchor="w")
        self.info3.grid(row=4, column=1,  sticky="nsew")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        # self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        
         # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        # self.grid_columnconfigure(0,minsize=500) #set width of first column
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew") #"nsew" có nghĩa là widget con sẽ giãn ra theo phương ngang và dọc của ô đặt của widget cha.
        self.sidebar_frame.grid_rowconfigure(7, weight=1) #5 is max row in 1 grid column

        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=avatar,text=" Bdobe Lightroom",compound="left", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_open_image = customtkinter.CTkButton(self.sidebar_frame, image=open,text="Open Image",command=self.open_file, text_color	 ="white")
        self.sidebar_button_open_image.grid(row=1, column=0, padx=20, pady=10)

        self.sidebar_button_negative = customtkinter.CTkButton(self.sidebar_frame, text="Negative",command=self.handle_negative,text_color	 ="white")
        self.sidebar_button_negative.grid(row=2, column=0, padx=20, pady=10)

        self.sidebar_button_sobel = customtkinter.CTkButton(self.sidebar_frame, text="Sobel Filter",command=self.handle_sobel,text_color	 ="white")
        self.sidebar_button_sobel.grid(row=3, column=0, padx=20, pady=10)

        self.sidebar_button_reset = customtkinter.CTkButton(self.sidebar_frame, image=reset,text="Reset Image",command=self.reset_image, text_color	 ="white")
        self.sidebar_button_reset.grid(row=4, column=0, padx=20, pady=10)
        
        # Draw Button
        self.sidebar_button_draw = customtkinter.CTkButton(self.sidebar_frame,image=draw,text="Draw",command=self.enable_drawing,text_color	 ="white")
        self.sidebar_button_draw.grid(row=5, column=0, padx=20, pady=10)

        # Export Edited Image
        self.sidebar_button_convolution = customtkinter.CTkButton(self.sidebar_frame, image=export,text="Export",command=self.export,text_color	 ="white")
        self.sidebar_button_convolution.grid(row=6, column=0, padx=20, pady=10)

        # Save Button
        self.sidebar_button_save = customtkinter.CTkButton(self.sidebar_frame, image=photo1,text="Save",command=self.saveImage,text_color	 ="white")
        self.sidebar_button_save.grid(row=7, column=0, padx=20, pady=10)
        
        
        # start from the next grid (row 7)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=8, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=9, column=0, padx=20, pady=(10, 10))

        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=10, column=0, padx=20, pady=(10, 0))

        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))
        
        # Image frame
        self.image_frame = customtkinter.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, padx=20, pady=(10,10))

        self.img_initial = cv2.imread("image/open_image.jpg")
        self.img_initial = cv2.cvtColor(self.img_initial, cv2.COLOR_BGR2RGB)

        # Resize the image to a fixed size
        img_height, img_width, _ = self.img_initial.shape
        if img_height > img_width:
            new_height = 640
            new_width = int((img_width / img_height) * new_height)
        else:
            new_width = 480
            new_height = int((img_height / img_width) * new_width)
        # self.original_img_size = (new_width, new_height)
        
        self.img_initial = Image.fromarray(self.img_initial)
        self.img_initial = ImageTk.PhotoImage(self.img_initial)

        self.original_img_lbl = tk.Label(self.image_frame,image=self.img_initial)
        self.original_img_lbl.grid(row=0,column=0,padx=20,pady=(10,10))




        # handle image frame
        self.handle_img_frame=customtkinter.CTkFrame(self,fg_color="transparent")
        
        self.handle_img_frame.grid(row=0,column=3,padx=20,pady=(10,10))

        self.optionmenu_1 = customtkinter.CTkOptionMenu(self.handle_img_frame, dynamic_resizing=False,
                                                values=["Brightness", "Contrast", "Noise", "Laplacian", "Gaussian Lowpass", "Gaussian Highpass", "Butterworth", "Erode", "Dilate"],
                                                command=self.switch_tab_event)
        self.optionmenu_1.grid(row=0, column=3, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.optionmenu_1.set("Select Filter")
        
        # Brightness Filter
        self.brightness_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.brightness_frame.grid(row=1, column=3, padx=20, pady=(10, 10))

        self.slider_Log = customtkinter.CTkSlider(self.brightness_frame, from_=0, to=90, number_of_steps=200, command=self.handle_log)
        self.slider_Log.set(25)

        # Contrast Filter
        self.contrast_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.contrast_frame.grid(row=1, column=3, padx=20, pady=(10, 10))
        
        self.slider_Gamma = customtkinter.CTkSlider(self.contrast_frame, from_=0, to=50, number_of_steps=200, command=self.handle_gamma)
        self.slider_Gamma.set(10)
        
        # Noise Filter
        self.noise_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.noise_frame.grid(row=1, column=3, padx=20, pady=(10, 10))
        
        self.slider_Median = customtkinter.CTkSlider(self.noise_frame, from_=0, to=30, number_of_steps=200, command=self.handle_noise)
        self.slider_Median.set(0)

        # Laplacian Filter
        self.laplacian_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.laplacian_frame.grid(row=1, column=3, padx=20, pady=(10, 10))

        self.slider_Laplacian = customtkinter.CTkSlider(self.laplacian_frame, from_=0, to=30, number_of_steps=200, command=self.handle_laplacian)
        self.slider_Laplacian.set(0)

        # Gaussian Lowpass Filter
        self.gaussian_lowpass_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.gaussian_lowpass_frame.grid(row=1, column=3, padx=20, pady=(10, 10))

        self.slider_gaussian_lowpass = customtkinter.CTkSlider(self.gaussian_lowpass_frame, from_=0, to=30, number_of_steps=200, command=self.handle_gaussian_lowpass)
        self.slider_gaussian_lowpass.set(0)

        # Gaussian Highpass Filter
        self.gaussian_highpass_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.gaussian_highpass_frame.grid(row=1, column=3, padx=20, pady=(10, 10))

        self.slider_gaussian_highpass = customtkinter.CTkSlider(self.gaussian_highpass_frame, from_=0, to=30, number_of_steps=200, command=self.handle_gaussian_highpass)
        self.slider_gaussian_highpass.set(0)


        # Butterworth Filter
        self.butterworth_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.butterworth_frame.grid(row=1, column=3, padx=20, pady=(10, 10))

        self.slider_butterworth = customtkinter.CTkSlider(self.butterworth_frame, from_=0, to=500, number_of_steps=200, command=self.handle_butterworth)
        self.slider_butterworth.set(0)

        # Erode Filter
        self.erode_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.erode_frame.grid(row=1, column=3, padx=20, pady=(10, 10))

        self.slider_erode = customtkinter.CTkSlider(self.erode_frame, from_=0, to=30, number_of_steps=200, command=self.handle_erode)
        self.slider_erode.set(0)

        # Morphology Filter
        self.morphology_frame = customtkinter.CTkFrame(self.handle_img_frame, fg_color="transparent")
        self.morphology_frame.grid(row=1, column=3, padx=20, pady=(10, 10))

        self.slider_morphology = customtkinter.CTkSlider(self.morphology_frame, from_=0, to=30, number_of_steps=200, command=self.handle_morphology)
        self.slider_morphology.set(0)

    

        
        #set default values
        self.appearance_mode_optionemenu.set("Dark")    
        self.scaling_optionemenu.set("100%")


        # Test function
        # Bind mouse events to the image label
        self.original_img_lbl.bind("<ButtonPress-1>", self.start_drawing)
        self.original_img_lbl.bind("<B1-Motion>", self.draw)
        self.original_img_lbl.bind("<ButtonRelease-1>", self.stop_drawing)

        # Initialize drawing variables
        self.drawing = False
        self.boundary_points = []
        self.image_stack = []  # Stack to store previous image states


    def handle_negative(self):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.negativeImage()
        else:            
            self.edit_boundary("negative")

    def handle_sobel(self):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.sobel()
        else:            
            self.edit_boundary("sobel")

    # Handle Slider
    def handle_log(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.log_Transform()
        else:            
            self.edit_boundary_slider(value, "brightness")

    def handle_gamma(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.gamma_image()
        else:            
            self.edit_boundary_slider(value, "gamma")

    def handle_noise(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.median_image()
        else:            
            self.edit_boundary_slider(value, "noise")

    def handle_laplacian(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.laplacian_image()
        else:            
            self.edit_boundary_slider(value, "laplacian")

    def handle_gaussian_lowpass(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.gaussian_lowpass()
        else:            
            self.edit_boundary_slider(value, "gaussian_lowpass")

    def handle_gaussian_highpass(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.gaussian_highpass()
        else:            
            self.edit_boundary_slider(value, "gaussian_highpass")

    def handle_butterworth(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.butterworth()
        else:            
            self.edit_boundary_slider(value, "butterworth")

    def handle_erode(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.erode()
        else:            
            self.edit_boundary_slider(value, "erode")

    def handle_morphology(self, value):
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.morphology()
        else:            
            self.edit_boundary_slider(value, "morphology")





    def enable_drawing(self):
        cancel_img=Image.open("image/cancel.png")
        cancel=ImageTk.PhotoImage(cancel_img)
        draw_img=Image.open("image/draw.png")
        draw=ImageTk.PhotoImage(draw_img)
        if self.sidebar_button_draw.cget("text") == "Draw":
            self.sidebar_button_draw.configure(text="Cancel",image=cancel)
            self.drawing = True
            self.original_img_lbl.configure(cursor="pencil")
            self.slider_Log.set(25)
            self.slider_Gamma.set(10)
            self.slider_Median.set(0)
            self.slider_Laplacian.set(0)
            self.slider_gaussian_lowpass.set(0)
            self.slider_gaussian_highpass.set(0)
            self.slider_butterworth.set(0)
            self.slider_erode.set(0)
            self.slider_morphology.set(0)
        else:
            self.sidebar_button_draw.configure(text="Draw",image=draw)
            self.drawing = False
            # Set the Slider back to 0
            current_option = self.optionmenu_1.get()
            if current_option == "Brightness":
                self.slider_Log.set(25)
            if current_option == "Contrast":
                self.slider_Gamma.set(10)
            if current_option == "Noise":
                self.slider_Median.set(0)
            if current_option == "Laplacian":
                self.slider_Laplacian.set(0)
            if current_option == "Gaussian Lowpass":
                self.slider_gaussian_lowpass.set(0)
            if current_option == "Gaussian Highpass":
                self.slider_gaussian_highpass.set(0)
            if current_option == "Butterworth":
                self.slider_butterworth.set(0)
            if current_option == "Erode":
                self.slider_erode.set(0)
            if current_option == "Dilate":
                self.slider_morphology.set(0)



    def start_drawing(self, event):
        if self.drawing:
            self.boundary_points = [(event.x, event.y)]

    def draw(self, event):
        if self.drawing:
            self.boundary_points.append((event.x, event.y))
            self.draw_boundary()

    def stop_drawing(self, event):
        self.drawing = False

    def draw_boundary(self):
        drawn_img = self.img_initial_copy.copy()
        draw = ImageDraw.Draw(drawn_img)
        draw.line(self.boundary_points, fill="red", width=2)

        drawn_img_tk = ImageTk.PhotoImage(drawn_img)
        self.original_img_lbl.configure(image=drawn_img_tk)
        self.original_img_lbl.image = drawn_img_tk

    def negativeImage_boundary(self, img):
        img = 255 -img
        return img

    def sobel_boundary(self, imgg):
        # imgg = self.temp_image
        if imgg.ndim != 2:
            imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
        ky=np.array([[-1.0,-2,-1],[0,0,0],[1,2,1]])
        kx=np.transpose(ky)
        Gx=self.Conv_Highpass(imgg,kx)
        Gy=self.Conv_Highpass(imgg,ky)
        Gm=np.sqrt(Gx**2+Gy**2)
        imgg = Gm
        return imgg
        # self.changed_image = Gm
        # self.t_image = self.changed_image
        # self.show_image()   

    def log_Transform_boundary(self, imgg, log_scale):
            #         self.changed_image = self.temp_image
            # img = np.array(self.changed_image, dtype='float')
            # img = img * (log_Scale / 25)  # Adjust the scaling factor
            # img = np.clip(img, 0, 255)  # Ensure the values are within the valid range
            # img = np.array(img, dtype='uint8')
            # self.changed_image = img

        c = log_scale
        log_image = np.array(imgg, dtype='float')
        log_image = log_image * (c/25)
        log_image = np.clip(log_image, 0, 255)
        # log_image = c * (np.log(imgg + 1))
        log_image = np.array(log_image, dtype='uint8')
        imgg = log_image
        return imgg
    
    def gamma_image_boundary(self, imgg, log_gamma):
        c = log_gamma
        if c != 0:
            
            imgg = np.array(255*(imgg/255)**(c/10),dtype='uint8')
            return imgg

    def conv_boundary(self,imgg,k):
        Out=np.zeros_like(imgg)
        if imgg.ndim >2:
            for i in range(3):
                Out[:,:,i]=convolve(imgg[:,:,i],k)
        else:
            Out=convolve(imgg,k)
        return Out

    def median_image_boundary(self, imgg, log_median):
        t = int(log_median)
        if t != 0:
            k = np.ones((t,t))/(t*t)
            imgg = self.conv_boundary(imgg,k)
            return imgg
            
    def laplacian_image_boundary(self, imgg, log_laplacian):
        t = int(log_laplacian)

        if t != 0:
            ksize=(t,t)
            imgg = cv2.blur(imgg, ksize, cv2.BORDER_DEFAULT)

            k2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
            imgg = self.Conv_Highpass(imgg,k2) + imgg
            return imgg

    def gaussian_lowpass_boundary(self, imgg, log_gaussian_lowpass):
        t = int(log_gaussian_lowpass)

        if t != 0:
            if imgg.ndim != 2:
                imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
            F = np.fft.fft2(imgg)
            F = np.fft.fftshift(F)
            M, N = imgg.shape
            D0 = t

            u = np.arange(0,M) - M/2; v = np.arange(0,N) - N/2
            [V,U] = np.meshgrid(v,u)
            D = np.sqrt(np.power(U,2) + np.power(V,2))
            H = np.array(D<=D0, 'float')
            G=H*F

            G=np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            imgg = imgOut
            return imgg
        
    def gaussian_highpass_boundary(self, imgg, log_gaussian_highpass):
        t = int(log_gaussian_highpass)

        if t != 0:
            if imgg.ndim != 2:
                imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
            F = np.fft.fft2(imgg)
            F = np.fft.fftshift(F)
            M, N = imgg.shape
            D0 = t

            u = np.arange(0,M) - M/2; v = np.arange(0,N) - N/2
            [V,U] = np.meshgrid(v,u)
            D = np.sqrt(np.power(U,2) + np.power(V,2))
            H = 1 - np.exp(-(D * D) / (2 * D0 * D0))
            G=H*F

            G=np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            imgg = imgOut
            return imgg


    def butterworth_boundary(self, imgg, log_butterworth):
        t = int(log_butterworth)

        if t != 0:
            # imgg = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
            F=np.fft.fft2(imgg)
            n=2; D0=t

            F=np.fft.fftshift(F)
            M, N = imgg.shape
            u = np.arange(0,M) - M/2; v = np.arange(0,N) - N/2
            [V,U] = np.meshgrid(v,u)
            D = np.sqrt(np.power(U,2) + np.power(V,2))
            H = 1/np.power(1 + (D/D0), (2*n))
            G=H*F

            G=np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            imgg = imgOut
            return imgg


    def erode_boundary(self, imgg, log_erode):
        t = int(log_erode)

        if t != 0:
            threshval = 100; n = 255;
            _, imgB = cv2.threshold(imgg, threshval, n, cv2.THRESH_BINARY)

            kernel1 = np.ones((t,t), np.uint8)
            img_erol1 = cv2.erode(imgB, kernel1, iterations=1)
            imgg = img_erol1
            return imgg
        
    def morphology_boundary(self, imgg, log_morphology):
        t = int(log_morphology)

        if t != 0:
            threshval = 100; n = 255
            _, imB = cv2.threshold(imgg, threshval, n, cv2.THRESH_BINARY)
            kernel = np.ones((t,t), np.uint8)
            img_dil = cv2.dilate(imB, kernel)
            imgg = img_dil
            return imgg

    def edit_boundary(self, button_type):
        boundary_mask = Image.new("L", self.img_initial_copy.size, 0)
        draw = ImageDraw.Draw(boundary_mask)
        draw.polygon(self.boundary_points, fill=255)

        boundary_mask_np = np.array(boundary_mask)
        img_np = np.array(self.img_initial_copy)

        if img_np.ndim == 2:  # Grayscale image
            if button_type == "negative":
                edited_img_np = np.where(boundary_mask_np, self.negativeImage_boundary(img_np), img_np)
            elif button_type == "sobel":
                edited_img_np = np.where(boundary_mask_np, self.sobel_boundary(img_np), img_np)
        elif img_np.ndim == 3 and img_np.shape[2] == 3:  # RGB image
            edited_img_np = img_np.copy()
            for channel in range(3):
                if button_type == "negative":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.negativeImage_boundary(img_np[..., channel]), img_np[..., channel])
                elif button_type == "sobel":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.sobel_boundary(img_np[..., channel]), img_np[..., channel])
        else:
            raise ValueError("Unsupported image format")

        self.t_image = edited_img_np
        edited_img = Image.fromarray(edited_img_np)

        edited_img_tk = ImageTk.PhotoImage(edited_img)
        self.original_img_lbl.configure(image=edited_img_tk)
        self.original_img_lbl.image = edited_img_tk  # Store a reference to avoid garbage collection

   
    def edit_boundary_slider(self, slider_value, slider_type):
        boundary_mask = Image.new("L", self.img_initial_copy.size, 0)
        draw = ImageDraw.Draw(boundary_mask)
        draw.polygon(self.boundary_points, fill=255)

        boundary_mask_np = np.array(boundary_mask)
        img_np = np.array(self.img_initial_copy)

        if img_np.ndim == 2:  # Grayscale image
            if slider_type == "gamma":
                edited_img_np = np.where(boundary_mask_np, self.gamma_image_boundary(img_np, slider_value), img_np)
            elif slider_type == "log":
                edited_img_np = np.where(boundary_mask_np, self.log_Transform_boundary(img_np, slider_value), img_np)
            elif slider_type == "noise":
                edited_img_np = np.where(boundary_mask_np, self.median_image_boundary(img_np, slider_value), img_np)
            elif slider_type == "laplacian":
                edited_img_np = np.where(boundary_mask_np, self.laplacian_image_boundary(img_np, slider_value), img_np)
            elif slider_type == "gaussian_lowpass":
                edited_img_np = np.where(boundary_mask_np, self.gaussian_lowpass_boundary(img_np, slider_value), img_np)
            elif slider_type == "gaussian_highpass":
                edited_img_np = np.where(boundary_mask_np, self.gaussian_highpass_boundary(img_np, slider_value), img_np)
            elif slider_type == "butterworth":
                edited_img_np = np.where(boundary_mask_np, self.butterworth_boundary(img_np, slider_value), img_np)
            elif slider_type == "erode":
                edited_img_np = np.where(boundary_mask_np, self.erode_boundary(img_np, slider_value), img_np)
            elif slider_type == "morphology":
                edited_img_np = np.where(boundary_mask_np, self.morphology_boundary(img_np, slider_value), img_np)

            else:
                raise ValueError("Unsupported slider type")
        elif img_np.ndim == 3 and img_np.shape[2] == 3:  # RGB image
            edited_img_np = img_np.copy()
            for channel in range(3):
                if slider_type == "gamma":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.gamma_image_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "brightness":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.log_Transform_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "noise":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.median_image_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "laplacian":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.laplacian_image_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "gaussian_lowpass":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.gaussian_lowpass_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "gaussian_highpass":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.gaussian_highpass_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "butterworth":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.butterworth_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "erode":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.erode_boundary(img_np[..., channel], slider_value), img_np[..., channel])
                elif slider_type == "morphology":
                    edited_img_np[..., channel] = np.where(boundary_mask_np, self.morphology_boundary(img_np[..., channel], slider_value), img_np[..., channel])

                else:
                    raise ValueError("Unsupported slider type")        
        else:
            raise ValueError("Unsupported image format")

        self.t_image = edited_img_np
        edited_img = Image.fromarray(edited_img_np)

        edited_img_tk = ImageTk.PhotoImage(edited_img)
        self.original_img_lbl.configure(image=edited_img_tk)
        self.original_img_lbl.image = edited_img_tk  # Store a reference to avoid garbage collection



    def negativeImage(self, *args):
        self.changed_image = self.temp_image
        self.changed_image = 255 - self.changed_image
        self.t_image = self.changed_image
        self.show_image()

    def log_Transform(self, *args):
        if self.img_path == None:
            return 0
        log_Scale = self.slider_Log.get()
        c = log_Scale
        if c != 25:
            self.changed_image = self.temp_image
            img = np.array(self.changed_image, dtype='float')
            img = img * (log_Scale / 25)  # Adjust the scaling factor
            img = np.clip(img, 0, 255)  # Ensure the values are within the valid range
            img = np.array(img, dtype='uint8')
            self.changed_image = img
        self.t_image = self.changed_image
        self.show_image()


    #main function
    
    def switch_tab_event(self, value):
        if value == "Brightness":
            self.brightness_frame.tkraise()
            self.contrast_frame.lower()
            self.noise_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.lower()
            self.erode_frame.lower()
            self.morphology_frame.lower()
            self.slider_Log.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        elif value == "Contrast":
            self.contrast_frame.tkraise()
            self.brightness_frame.lower()
            self.noise_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.lower()
            self.erode_frame.lower()
            self.morphology_frame.lower()
            self.slider_Gamma.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        elif value == "Noise":
            self.noise_frame.tkraise()
            self.brightness_frame.lower()
            self.contrast_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.lower()
            self.erode_frame.lower()
            self.morphology_frame.lower()
            self.slider_Median.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        elif value == "Laplacian":
            self.noise_frame.lower()
            self.brightness_frame.lower()
            self.contrast_frame.lower()
            self.laplacian_frame.tkraise()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.lower()
            self.erode_frame.lower()
            self.morphology_frame.lower()
            self.slider_Laplacian.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        elif value == "Gaussian Lowpass":
            self.noise_frame.lower()
            self.brightness_frame.lower()
            self.contrast_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.tkraise()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.lower()
            self.erode_frame.lower()
            self.morphology_frame.lower()
            self.slider_gaussian_lowpass.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")
        elif value == "Gaussian Highpass":
            self.noise_frame.lower()
            self.brightness_frame.lower()
            self.contrast_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.tkraise()
            self.butterworth_frame.lower()
            self.erode_frame.lower()
            self.morphology_frame.lower()
            self.slider_gaussian_highpass.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")


        elif value == "Butterworth":
            self.noise_frame.lower()
            self.brightness_frame.lower()
            self.contrast_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.tkraise()
            self.erode_frame.lower()
            self.morphology_frame.lower()
            self.slider_butterworth.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        elif value == "Erode":
            self.noise_frame.lower()
            self.brightness_frame.lower()
            self.contrast_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.lower()
            self.erode_frame.tkraise()
            self.morphology_frame.lower()
            self.slider_erode.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")

        elif value == "Dilate":
            self.noise_frame.lower()
            self.brightness_frame.lower()
            self.contrast_frame.lower()
            self.laplacian_frame.lower()
            self.gaussian_lowpass_frame.lower()
            self.gaussian_highpass_frame.lower()
            self.butterworth_frame.lower()
            self.erode_frame.lower()
            self.morphology_frame.tkraise()
            self.slider_morphology.grid(row=3, column=0, padx=(20, 10), pady=(10, 10), sticky="ew")



    def sobel(self, *args):
        img = self.temp_image
        if img.ndim != 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ky=np.array([[-1.0,-2,-1],[0,0,0],[1,2,1]])
        kx=np.transpose(ky)
        Gx=self.Conv_Highpass(img,kx)
        Gy=self.Conv_Highpass(img,ky)
        Gm=np.sqrt(Gx**2+Gy**2)
        self.changed_image = Gm
        self.t_image = self.changed_image
        self.show_image()   
        
    #support function
    
    def sidebar_button_event(self):
        print("sidebar_button click")   

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100

        customtkinter.set_widget_scaling(new_scaling_float)

    # Resize Image 
    def resize_image(self,img, width, height):
        if img is None:
            return None

        img_height, img_width = img.shape[:2]

        # Calculate the aspect ratio of the image
        aspect_ratio = img_width / img_height

        if width is not None and height is not None:
            # Resize the image to the specified width and height
            resized_img = cv2.resize(img, (width, height))
        elif width is not None:
            # Calculate the new height based on the aspect ratio and the desired width
            new_height = int(width / aspect_ratio)
            resized_img = cv2.resize(img, (width, new_height))
        elif height is not None:
            # Calculate the new width based on the aspect ratio and the desired height
            new_width = int(height * aspect_ratio)
            resized_img = cv2.resize(img, (new_width, height))
        else:
            # Return None if no width or height is specified
            return None

        return resized_img



    def show_image(self, *args):
        self.changed_image = self.resize_image(self.changed_image,640, 480)
        self.changed_image = Image.fromarray(self.changed_image)
        self.img_initial_copy = self.changed_image.copy()
        self.changed_image = ImageTk.PhotoImage(self.changed_image)
        self.original_img_lbl.configure(image=self.changed_image)
        self.original_img_lbl.image = self.changed_image


    def open_file(self):
        global once
        once = True
        img_file = filedialog.askopenfilename() 
        if img_file  != '':     
            self.img_path = img_file 
            # self.log_Scale.set(self.log_Scale.get() + 1)
            # self.log_Scale.set(self.log_Scale.get() - 1)

            self.original_image = cv2.imread(self.img_path)
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.changed_image = self.original_image
            self.temp_image = self.changed_image
            self.show_image()
        else:
            return 0

    def conv(self,A,k):
        # self.original_image = cv2.imread(self.img_path,cv2.IMREAD_COLOR)
        # self.changed_image = self.original_image
        self.changed_image = self.temp_image
        Out=np.zeros_like(self.changed_image)
        if self.changed_image.ndim >2:
            for i in range(3):
                Out[:,:,i]=convolve(self.changed_image[:,:,i],k)
        else:
            Out=convolve(self.changed_image,k)
        return Out

    def Conv_Highpass(self, img,k):
        Input=np.array(img,dtype='single')
        if Input.ndim >2:
            Out=np.zeros_like(Input)
            for i in range(3):
                Out[:,:,i]=np.convolve(Input[:,:,i],k)
        else:
            Out=convolve(Input,k)
        return Out


    #main function

    def gamma_image(self, *args):
        if self.img_path == None:
            return 0
        log_gamma = self.slider_Gamma.get()
        c = log_gamma
        if c != 0:
        # self.original_image = cv2.imread(self.img_path,cv2.IMREAD_COLOR)
            self.changed_image = self.temp_image
            
            self.changed_image = np.array(255*(self.changed_image/255)**(c/10),dtype='uint8')
            self.t_image = self.changed_image
            self.show_image()

    def median_image(self, *args):
        t = int(self.slider_Median.get())
        if t != 0:
            k = np.ones((t,t))/(t*t)
            self.changed_image = self.temp_image
            self.changed_image = self.conv(self.changed_image,k)
            self.t_image = self.changed_image
            self.show_image()
            

    def laplacian_image(self, *args):
        if self.img_path == None:
            return 0

        t = int(self.slider_Laplacian.get())

        if t != 0:
            # self.original_image = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
            self.original_image = self.temp_image
            if self.original_image.ndim != 2:
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            # self.changed_image = self.original_image
            ksize=(t,t)
            self.original_image = cv2.blur(self.original_image, ksize, cv2.BORDER_DEFAULT)

            # k1=np.array([[0,1,0],[1,-4,1],[0,1,0]])
            k2=np.array([[1,1,1],[1,-8,1],[1,1,1]])
            self.original_image = self.Conv_Highpass(self.original_image,k2)+self.original_image

            self.changed_image = self.original_image
            self.t_image = self.changed_image

            self.show_image()

    def gaussian_lowpass(self, *args):
        if self.img_path is None:
            return 0

        t = int(self.slider_gaussian_lowpass.get())

        if t != 0:
            img = self.temp_image

            F = np.fft.fft2(img, axes=(0, 1))  # Apply FFT separately on each color channel
            F = np.fft.fftshift(F, axes=(0, 1))
            M, N, C = img.shape
            D0 = t

            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt(np.power(U, 2) + np.power(V, 2))
            H = np.array(D <= D0, dtype='float')
            H = np.stack([H] * C, axis=2)  # Replicate the filter for each color channel

            G = H * F

            G = np.fft.ifftshift(G, axes=(0, 1))
            imgOut = np.real(np.fft.ifft2(G, axes=(0, 1)))

            imgOut = np.abs(imgOut)  # Ensure the values are non-negative
            imgOut = np.clip(imgOut, 0, 255).astype('uint8')  # Clip values to valid range [0, 255]

            self.changed_image = imgOut
            self.t_image = self.changed_image
            self.show_image()

    def gaussian_highpass(self, *args):
        if self.img_path == None:
            return 0

        t = int(self.slider_gaussian_highpass.get())

        if t != 0:
            img = self.temp_image

            F = np.fft.fft2(img, axes=(0, 1))  # Apply FFT separately on each color channel
            F = np.fft.fftshift(F, axes=(0, 1))
            M, N, C = img.shape
            D0 = t

            u = np.arange(0, M) - M / 2
            v = np.arange(0, N) - N / 2
            [V, U] = np.meshgrid(v, u)
            D = np.sqrt(np.power(U, 2) + np.power(V, 2))
            H = 1 - np.exp(-(D * D) / (2 * D0 * D0))
            H = np.stack([H] * C, axis=2)  # Replicate the filter for each color channel

            G = H * F

            G = np.fft.ifftshift(G, axes=(0, 1))
            imgOut = np.real(np.fft.ifft2(G, axes=(0, 1)))

            imgOut = np.abs(imgOut)  # Ensure the values are non-negative
            imgOut = np.clip(imgOut, 0, 255).astype('uint8')  # Clip values to valid range [0, 255]

            self.changed_image = imgOut
            self.t_image = self.changed_image
            self.show_image()


    def butterworth(self, *args):
        if self.img_path == None:
            return 0
        t = int(self.slider_butterworth.get())
        if t != 0:
            img = self.temp_image
            if img.ndim != 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            F=np.fft.fft2(img)
            n=2; D0=t
            F=np.fft.fftshift(F)
            M, N = img.shape
            u = np.arange(0,M) - M/2; v = np.arange(0,N) - N/2
            [V,U] = np.meshgrid(v,u)
            D = np.sqrt(np.power(U,2) + np.power(V,2))
            H = 1/np.power(1 + (D/D0), (2*n))
            G=H*F
            G=np.fft.ifftshift(G)
            imgOut = np.real(np.fft.ifft2(G))
            self.changed_image = imgOut
            self.t_image = self.changed_image
            self.show_image()

    def erode(self, *args):
        if self.img_path == None:
            return 0
        t = int(self.slider_erode.get())
        if t != 0:
            img = self.temp_image
            threshval = 100; n = 255;
            retval, imgB = cv2.threshold(img, threshval, n, cv2.THRESH_BINARY)
            kernel1 = np.ones((t,t), np.uint8)
            img_erol1 = cv2.erode(imgB, kernel1, iterations=1)
            self.changed_image = img_erol1
            self.t_image = self.changed_image
            self.show_image()

    def morphology(self, *args):
        if self.img_path == None:
            return 0
        t = int(self.slider_morphology.get())
        if t != 0:
            img = self.temp_image
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            threshval = 100; n = 255
            _, imB = cv2.threshold(img, threshval, n, cv2.THRESH_BINARY)
            kernel = np.ones((t,t), np.uint8)
            img_dil = cv2.dilate(imB, kernel)
            self.changed_image = img_dil
            self.t_image = self.changed_image
            self.show_image()

    def export(self):
    # Define the file types and extensions
        filetypes = [("JPEG", "*.jpg"), ("PNG", "*.png")]

        # Open a file dialog to choose the save location and file type
        file_path = tkinter.filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=filetypes)

        if file_path:
            # Convert the PhotoImage to a PIL Image
            pil_image = self.t_image

            # Convert the image to RGB mode if it has an alpha channel
            if pil_image.mode == "RGBA":
                pil_image = pil_image.convert("RGB")

            # Determine the file format based on the chosen file extension
            file_extension = file_path.split(".")[-1].lower()
            if file_extension == "jpg" or file_extension == "jpeg":
                file_format = "JPEG"
            elif file_extension == "png":
                file_format = "PNG"
            else:
                print("Invalid file format. Export canceled.")
                return

            # Save the PIL Image to the chosen file path with the selected format
            pil_image.save(file_path, format=file_format)
            print("Image saved successfully.")
        else:
            print("Export canceled.")

            
    def saveImage(self, *args):
        self.temp_image = self.t_image
        self.changed_image = self.t_image
        self.t_image = self.resize_image(self.t_image,640, 480)
        self.t_image = Image.fromarray(self.t_image)
        self.img_initial_copy = self.t_image.copy()

        self.sidebar_button_draw.configure(text="Draw")
        self.drawing = False


    def reset_image(self,*args):

        self.original_image = cv2.imread(self.img_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.changed_image = self.original_image
        self.t_image = self.changed_image
        self.temp_image = self.changed_image
        self.original_image = Image.fromarray(self.original_image)
        self.original_image = ImageTk.PhotoImage(self.original_image)
        self.original_img_lbl.configure(image=self.original_image)
        self.original_img_lbl.image = self.original_image
        self.show_image()
        
        self.slider_Log.set(25)
        self.slider_Gamma.set(10)
        self.slider_Median.set(0)
        self.slider_Laplacian.set(0)
        self.slider_gaussian_lowpass.set(0)
        self.slider_gaussian_highpass.set(0)
        self.slider_butterworth.set(0)
        self.slider_erode.set(0)
        self.slider_morphology.set(0)
        self.sidebar_button_draw.configure(text="Draw")
        self.drawing = False


if __name__ == "__main__":
    app = App()
    app.mainloop()



# %%
