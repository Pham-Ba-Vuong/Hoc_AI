# I OPENCV
- 1.cbi
``` Python
import cv2 # nạp thư viện OpenCV, dùng cho mọi thao tác xử lý ảnh.
import numpy as np # nạp NumPy. Ảnh trong OpenCV thực chất là mảng NumPy.
import matplotlib.pyplot as plt # thư viện vẽ và hiển thị ảnh trong notebook.
from google.colab import files # công cụ upload và download file của Colab.

# Danh sách ảnh đã upload. Chạy lại ô này KHÔNG làm mất ảnh cũ.
ds_anh = globals().get("ds_anh", {})

def show(images, titles, cols=3, size=4):
    """Hiển thị nhiều ảnh cạnh nhau."""
    rows = (len(images) + cols - 1) // cols #tính số hàng bằng phép chia làm tròn lên. 
    plt.figure(figsize=(size * cols, size * rows)) # tạo khung vẽ có kích thước (rộng, cao) theo inch.
    for i, (im, t) in enumerate(zip(images, titles)): # duyệt đồng thời từng ảnh và tiêu đề của nó
        plt.subplot(rows, cols, i + 1) # chọn ô thứ i+1 trong lưới
        if im.ndim == 2: 
            plt.imshow(im, cmap="gray") # hiển thị ảnh xám bằng bảng màu xám (nếu không có cmap matplotlib sẽ tô màu giả).
        else:
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) # ảnh màu phải đổi BGR sang RGB, nếu không màu bị đảo
        plt.title(t) 
        plt.axis("off") 
    plt.tight_layout() # tự căn khoảng cách để các ô không đè lên nhau.
    plt.show() # vẽ ra màn hình.

def thu_nho(im, max_w=800):
    """Thu nhỏ ảnh lớn để chạy nhanh hơn."""
    h, w = im.shape[:2] # shape là (cao, rộng, kênh), [:2] lấy hai giá trị đầu.
    if w > max_w:
        im = cv2.resize(im, (max_w, int(h * max_w / w)))
    return im # trả ảnh

print("San sang!")
```
- 2.upload ảnh
``` Python
uploaded = files.upload()   # bấm "Chọn tệp", giữ Ctrl/Shift để chọn nhiều ảnh

for ten, data in uploaded.items():
    # Đọc từ dữ liệu trực tiếp, tên file tiếng Việt có dấu vẫn không lỗi
    im = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print(f"Bo qua '{ten}' (khong phai file anh)")
        continue
    ds_anh[ten] = thu_nho(im) # thu nhỏ ảnh rồi lưu vào từ điển với khóa là tên file.

ds_ten = list(ds_anh.keys()) # lấy danh sách tên để đánh số thứ tự.
print(f"\nHien co {len(ds_ten)} anh:")
for i, ten in enumerate(ds_ten):
    print(f"  [{i}] {ten}  {ds_anh[ten].shape}")

show(list(ds_anh.values()), [f"[{i}] {t[:25]}" for i, t in enumerate(ds_ten)], cols=3) # hiển thị mọi ảnh. .values() lấy các ảnh, t[:25] cắt tên còn 25 ký tự cho tiêu đề không quá dài.
```
- 3.chọn ảnh để xử lí
```Python
chon = 0   # số thứ tự ảnh muốn lấy

ds_ten = list(ds_anh.keys()) # cập nhật danh sách tên (phòng khi vừa thêm ảnh).
if chon >= len(ds_ten):
    raise IndexError(f"Chi co {len(ds_ten)} anh (so 0 den {len(ds_ten)-1})")

img = ds_anh[ds_ten[chon]] # ds_ten[chon] ra tên, ds_anh[tên] ra ảnh. Biến img này được mọi phần sau sử dụng. 
print("Dang xu ly:", ds_ten[chon], img.shape)
show([img], ["Anh dang chon"], cols=1, size=5)
```
## 1. thay đổi ko gian màu
- Tách rời thông tin độ sáng va màu sắc
- Đơn giản hóa và tăng tốc độ xử lý
- Tách/Lọc màu hiệu quả hơn (HSV / HSL)
- - Đối với HSV, phạm vi màu sắc là [0,179], phạm vi độ bão hòa là [0,255] và phạm vi độ sáng là [0,255]. 
- Cân bằng và cải thiện chất lượng ảnh
```Python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # cvtColor là hàm đổi không gian màu, mã BGR2GRAY đổi sang ảnh xám 1 kênh.
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # đổi sang HSV (H: sắc độ, S: độ bão hòa, V: độ sáng). HSV tách màu khỏi độ sáng nên lọc màu ổn định hơn BGR.

# Lọc màu theo HSV (ví dụ xanh dương). H: 0-179, S và V: 0-255
lower = np.array([100, 80, 50])
upper = np.array([130, 255, 255])

mask   = cv2.inRange(hsv, lower, upper) # tạo mặt nạ, pixel nào nằm trong khoảng thì bằng 255 (trắng), còn lại 0 (đen).
result = cv2.bitwise_and(img, img, mask=mask) # phép AND ảnh với chính nó, chỉ giữ pixel nơi mask trắng, phần còn lại thành đen.

show([img, gray, hsv, mask, result],
     ["Goc", "Xam", "HSV (hien thi)", "Mask xanh duong", "Ket qua loc mau"], cols=3)

# Tra giá trị HSV của một màu BGR
bgr = np.uint8([[[0, 255, 0]]]) # tạo ảnh 1X1 pixel xanh lá
print("HSV cua xanh la:", cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]) #: đổi sang HSV rồi lấy pixel duy nhất.
```
## 2.Biến đổi hình học của ảnh
-  Hiệu chỉnh góc nhìn và loại bỏ biến dạng phối cảnh
```Python
h, w = img.shape[:2] # lấy cao,rộng

nho = cv2.resize(img, (w // 2, h // 2)) # thu nho 1 nửa
to  = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC) # phóng 1.5 lần

M = np.float32([[1, 0, 100], [0, 1, 50]]) # tịnh tiến,phải 100px, xuống 50px
dich = cv2.warpAffine(img, M, (w, h)) # áp ma trận lên ảnh là ảnh kích thước đầu ra 

M = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1.0)  # xoay 45 độ
xoay = cv2.warpAffine(img, M, (w, h))

lat = cv2.flip(img, 1)                                  # lật ngang , 0 là lật dọc , -1 lật cả 2

pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])# phối cảnh 4 góc ảnh (trái-trên, phải-trên, trái-dưới, phải-dưới)
pts2 = np.float32([[w*0.15, h*0.1], [w*0.85, 0], [0, h*0.9], [w, h]]) # vị trí mới 
M = cv2.getPerspectiveTransform(pts1, pts2) # biến đổi điểm 1->2
phoicanh = cv2.warpPerspective(img, M, (w, h)) # áp phép biến đổi phối cảnh

show([nho, to, dich, xoay, lat, phoicanh],
     [f"Nho {nho.shape[1]}x{nho.shape[0]}", f"To {to.shape[1]}x{to.shape[0]}",
      "Tinh tien", "Xoay 45 do", "Lat ngang", "Phoi canh"], cols=3)
```
## 3.Phân ngưỡng: chuyển ảnh xám thành ảnh nhị phân
- Tách đối tượng khỏi nền
- Giảm đáng kể dữ liệu xử lí
- Chuẩn hóa dữ liệu cho các thuật toán trích xuất đặc trưng
### 3.1.Ngưỡng đơn giản
- Hàm ` cv.threshold` được sử dụng để áp dụng ngưỡng
- - Tham số đầu tiên là ảnh nguồn, phải là ảnh xám .
- - Tham số thứ hai là giá trị ngưỡng được sử dụng để phân loại giá trị pixel.
- - Tham số thứ ba là giá trị tối đa được gán cho các giá trị pixel vượt quá ngưỡng. 
- - OpenCV cung cấp các loại ngưỡng khác nhau được thể hiện bởi tham số thứ tư của hàm.
- Tất cả loại ngưỡng cơ bản:
- - cv.THRESH_BINARY : nếu giá trị pixel lớn hơn ngưỡng, đặt thành maxval, ngược lại =0.
- - cv.THRESH_BINARY_INV: lớn hơn ngưỡng =0, ngc lại maxval.
- - cv.THRESH_TRUNC : lớn hơn ngưỡng thì giữ lại ở ngưỡng ,ngược lại đc giữ nguyên.
- - cv.THRESH_TOZERO: lớn hơn ngưỡng thì giữ nguyên , ngc lại =0
- - cv..THRESH_TOZERO_INV: lớn hơn ngưỡng =0, ngc lại giữ nguyên
### 3.2.Ngưỡng thích ứng
- xác định ngưỡng cho một pixel dựa trên một vùng nhỏ xung quanh nó.
- - cv.ADAPTIVE_THRESH_MEAN_C : Giá trị ngưỡng là giá trị trung bình của vùng lân cận trừ đi hằng số C.
- - cv.ADAPTIVE_THRESH_GAUSSIAN_C : Giá trị ngưỡng là tổng có trọng số Gaussian của các giá trị lân cận trừ đi hằng số C.(là một hằng số được trừ đi từ giá trị trung bình hoặc tổng có trọng số của các pixel trong vùng lân cận.)
### 3.2 sư nhị phân otsu
- tránh được việc phải chọn giá trị và tự động xác định giá trị đó.
- xác định giá trị ngưỡng toàn cục tối ưu từ biểu đồ tần suất của hình ảnh.
```Python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # đổi sang ảnh xám
blur = cv2.GaussianBlur(gray, (5, 5), 0) # làm mờ bằng kernel Gauss 5x5. Số 0 nghĩa là để OpenCV tự tính sigma. Giảm nhiễu trước khi phân ngưỡng

_, th_global = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) # pixel > 127 thành 255, còn lại thành 0. Hàm trả về hai giá trị (ngưỡng, ảnh), dấu _ bỏ qua giá trị ngưỡng.
th_adapt = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)# ngưỡng được tính riêng cho từng vùng: 255: giá trị khi đạt ngưỡng,ADAPTIVE_THRESH_GAUSSIAN_C: ngưỡng là trung bình có trọng số Gauss của vùng lân cận,11: kích thước vùng lân cận (phải là số lẻ),2: hằng số trừ khỏi trung bình.
ret, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # cộng cờ THRESH_OTSU để thuật toán Otsu tự tìm ngưỡng tối ưu. Số 0 truyền vào bị bỏ qua, và ret chính là ngưỡng Otsu tìm được.
print("Nguong Otsu tu tim duoc:", ret)

show([gray, th_global, th_adapt, th_otsu],
     ["Anh xam", "Toan cuc (127)", "Thich ung", f"Otsu ({ret:.0f})"],
     cols=4, size=3.5) # làm tròn 
```
## 4.làm mịn
- khử nhiễu
- Làm bước đệm bắt buộc trước khi phát hiện biên
- Làm mờ chi tiết rườm rà, tập trung vào cấu trúc tổng thể
```Python
# Tạo nhiễu muối tiêu để thấy rõ khác biệt giữa các bộ lọc
noisy = img.copy() # sao chép để không làm hỏng ảnh gốc.
n = int(0.02 * img.shape[0] * img.shape[1]) # số điểm nhiễu = 2 phần trăm tổng số pixel
# sinh n tọa độ x,y ngẫu nhiên
ys = np.random.randint(0, img.shape[0], n)
xs = np.random.randint(0, img.shape[1], n)
noisy[ys[:n//2], xs[:n//2]] = 255 # nửa đầu gán trắng
noisy[ys[n//2:], xs[n//2:]] = 0 # nửa sau gán đen

trung_binh  = cv2.blur(noisy, (5, 5)) # lọc trung bình, mỗi pixel thành trung bình 5x5 lân cận. Làm mờ cả cạnh.
gauss       = cv2.GaussianBlur(noisy, (5, 5), 0) # lọc Gauss, pixel gần tâm có trọng số lớn hơn nên mờ tự nhiên hơn
trung_vi    = cv2.medianBlur(noisy, 5) # lấy trung vị của vùng 5x5. Các điểm trắng hay đen cực đoan bị loại hẳn, loại bỏ nhiễu dạng hạt
song_phuong = cv2.bilateralFilter(noisy, 9, 75, 75) #lọc song phương, làm mịn nhưng giữ cạnh. 9 là đường kính vùng, hai số 75 là độ lệch theo màu và theo khoảng cách (càng lớn càng mờ).

show([noisy, trung_binh, gauss, trung_vi, song_phuong],
     ["Anh nhieu", "Trung binh", "Gauss", "Trung vi (tot nhat)", "Song phuong (giu bien)"],
     cols=3)
```  
## 5. Biến đổi hình thái
- Xói mòn:
- - Làm mỏng đối tượng, thu nhỏ kích thước vùng sáng (hoặc vùng có giá trị lớn), loại bỏ các điểm nhiễu nhỏ đơn lẻ (nhiễu muối tiêu dạng sáng) và tách các đối tượng bị dính liền nhau thành các phần riêng biệt.
- Giãn nở:
- - : Làm dày đối tượng, mở rộng vùng sáng, lấp đầy các lỗ hổng nhỏ bên trong đối tượng và kết nối các đoạn bị đứt gãy lại với nhau.
- phép Mở (Opening - Co rồi Giãn):
- - Giúp loại bỏ nhiễu nền ở bên ngoài (các điểm sáng nhỏ nằm độc lập) mà không làm thay đổi đáng kể kích thước hoặc hình dạng của đối tượng chính. Rất hữu ích khi cần xóa nhiễu nền mà vẫn giữ nguyên hình khối vật thể.
- phép Đóng (Closing - Giãn rồi Co):
- - Giúp lấp đầy các lỗ hổng bên trong đối tượng hoặc nối các khoảng hở nhỏ trên đường viền mà không làm thay đổi tổng thể kích thước của vật thể.
```Python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #tạo ảnh nhị phân
_, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((5, 5), np.uint8)

xoi_mon  = cv2.erode(bin_img, kernel, iterations=1)                 # xói mòn.
gian_no  = cv2.dilate(bin_img, kernel, iterations=1)                # giãn nở
mo_rong  = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)        # mở (xói mòn rồi giãn nở): bỏ nhiễu trắng nhỏ
dong_lai = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)       # đóng (giãn nở rồi xói mòn): lấp lỗ đen nhỏ
gradient = cv2.morphologyEx(bin_img, cv2.MORPH_GRADIENT, kernel)    # viền ngoài của vật thể

show([bin_img, xoi_mon, gian_no, mo_rong, dong_lai, gradient],
     ["Anh nhi phan", "Xoi mon", "Gian no", "Mo (Opening)", "Dong (Closing)", "Gradient hinh thai"],
     cols=3)
```
## 6.Độ dốc hình ảnh
- Phát hiện biên và cạnh
-  Phân đoạn ảnh
```Python
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sobel_x   = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)   # cạnh dọc:đạo hàm theo hướng x (dx=1, dy=0), làm nổi cạnh dọc. ksize=5 là kích thước kernel.
sobel_y   = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)   # cạnh ngang:đạo hàm theo hướng y, làm nổi cạnh ngang.

# Chuyển về 0-255 để hiển thị (lấy giá trị tuyệt đối vì độ dốc có thể âm)
lap = cv2.convertScaleAbs(laplacian)
sx  = cv2.convertScaleAbs(sobel_x)
sy  = cv2.convertScaleAbs(sobel_y)

show([gray, lap, sx, sy],
     ["Anh xam", "Laplacian", "Sobel X (canh doc)", "Sobel Y (canh ngang)"],
     cols=4, size=3.5)
```
## 7.Phát hiện cạnh cany
- Khử nhiễu hiệu quả trước khi xử lý
- Xác định đường biên mỏng và sắc nét
```Python
ray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# hai số là minVal và maxVal. Độ dốc > maxVal: chắc chắn là cạnh. Độ dốc < minVal: bỏ.
canny_thap = cv2.Canny(gray, 50, 100)     # ngưỡng thấp: nhiều cạnh, nhiều nhiễu
canny_vua  = cv2.Canny(gray, 100, 200)
canny_cao  = cv2.Canny(gray, 150, 300)    # ngưỡng cao: ít cạnh, sạch hơn

# Làm mờ trước giúp bớt cạnh nhiễu
canny_blur = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 100, 200)

show([gray, canny_thap, canny_vua, canny_cao, canny_blur],
     ["Anh xam", "Canny 50-100", "Canny 100-200", "Canny 150-300", "Lam mo + Canny 100-200"],
     cols=3)
```  
## 8.Kim tự tháp hình ảnh
- Kim tự tháp Gaussian:Dùng để thu nhỏ ảnh dần dần (giảm độ phân giải thông qua lọc làm mờ và lấy mẫu xuống)
- Kim tự tháp Laplacian:Dùng để lưu trữ thông tin chi tiết (các tần số cao) bị mất đi giữa các tầng, cho phép khôi phục lại ảnh gốc hoặc phục vụ các thuật toán ghép ảnh.
- tác dụng
- - Xử lý và phát hiện đối tượng đa tỷ lệ
- - Ghép ảnh toàn cảnh và trộn ảnh mượt mà
- - Tối ưu hóa tốc độ tính toán
- - Nén và truyền tải dữ liệu ảnh phân cấp
```Python
# Kim tự tháp Gauss: thu nhỏ dần (mỗi tầng còn 1/4 diện tích)
tang = [img]
for i in range(3):
    tang.append(cv2.pyrDown(tang[-1]))

show(tang, [f"Tang {i}: {t.shape[1]}x{t.shape[0]}" for i, t in enumerate(tang)], cols=4)

# Kim tự tháp Laplacian: ảnh gốc trừ ảnh (thu nhỏ rồi phóng lại)
nho = cv2.pyrDown(img)
to_lai = cv2.pyrUp(nho, dstsize=(img.shape[1], img.shape[0]))
lap_pyr = cv2.subtract(img, to_lai)
show([img, to_lai, lap_pyr * 5],
     ["Anh goc", "Thu nho roi phong lai (bi mat chi tiet)", "Chi tiet bi mat (nhan 5 cho de thay)"], cols=3)
```



   




 

