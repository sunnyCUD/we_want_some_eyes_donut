# Galucoma Project
This project is a part of image processing class project at Mahidol University 
## Getting Started
This folder contains 3 subfolders
```
1. Final program: The final program with simple GUI
2. Individual code: This folder contain subfolders of each member codes
3. RAW_FUNDUS_INPUT: This folder contains raw retina fundus images
```
### Installing
The final program requires `numpy`, `pandas`, `xlrd`, `glob`, `cv2`, `matplotlib`,
`pywt`, `scipy`, `tkinter`, `PIL`, `math`,and `skimage`

Some individual codes require more modules than this.

Some individual codes requires RAW_FUNDUS_INPUT folder in their directory.
### Running
1. Open Final program folder
2. Run Glaucoma_CDR_finder_program.ipynb
3. Execute every cells
4. The program GUI will pop up, then browse and select fundus image in RAW_FUNDUS_INPUT
5. The result will display
## Built with
* Jupyter Notebook
* Spyder
* VScode
* Matlab
## Known bugs
* OD localization tends to error when too much light leaks through the fundus camera
* If the input image size is changed, the localize program will crash
* If OD center is too clsoe to the image edge that cause size of ROI too change,
the cup segmentation program will crash.
* Cup segmentation program still has significant error which also cause error to CDR result.
## Authors
- **เกียรติพงศ์ สารวิทย์** - *initial work*
- **คริษฐา บูญจูงเกียรติ** - *initial work*
- **ชญานนท์ จงมีความสุข** - *initial work*
- **ชวนากร ปิยภาณีรัตน** - *initial work*
- **ชวัลชาติ เหราบัตย** - *initial work*
- **พีรณัฐ เหล่าธนสิน** - *initial work*
- **สฤษฎ์พงศ์ เพิ่มผล** - *initial work*
- **สิทธิกร ปุญญกริยากร** - *initial work*
- **ชมบุญ สีมารักษ** - *initial work*