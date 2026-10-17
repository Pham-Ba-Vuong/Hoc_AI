# I.CƠ BẢN VỀ NEURAL NETWORK
## 1.1 Cấu trúc perceptron
- Sơ đồ có lớp đầu vào (Input), lớp đầu ra (Output), các đầu vào x₁, x₂, …, x_d và một đầu vào bias x₀ = 1. Các kết nối có trọng số w₀, w₁, w₂, …, w_d; đầu ra được ký hiệu y.
- - Bias là một đầu vào cố định, giúp mô hình dịch chuyển ngưỡng phân loại thay vì buộc đường phân loại đi qua gốc tọa độ.
- - Perceptron được xem là cấu trúc một lớp do lớp đầu vào không thực hiện phép tính.
- Lớp đầu ra có một nút: tính tổng s = w₀x₀ + w₁x₁ + … + w_d·x_d, rồi đưa qua hàm kích hoạt ngưỡng (step/threshold): s ≥ 0 cho 1, ngược lại cho −1.
- -  Nếu tổng vượt ngưỡng thì nơ-ron “bật”; nếu không thì “tắt”. 
 - Có 3 hàm logic nhị phân chính :
- - AND: chỉ đúng khi cả hai đầu vào đều đúng. 
- - OR: trả về đúng nếu ít nhất một đầu vào bằng 1.  
- - XOR: đầu ra đúng khi hai mệnh đề có giá trị khác nhau
- 2 thuật toán phân loại nhị phân:
- - Biểu diễn trên mặt phẳng: Đường thẳng là ranh giới quyết định. Một phía được phân loại là −1, phía còn lại là +1.
- - Phân loại tuyến tính : Đường thẳng là ranh giới quyết định. Một phía được phân loại là −1, phía còn lại là +1.
- - - Perceptron phân loại đúng 100% tập huấn luyện với AND,OR
- - - Perceptron đơn chỉ đạt tối đa 75% độ chính xác với XOR => dùng MLP.
## 1.2. Multilayer Perceptron (MLP)
- tức là thêm lớp trung gian để biến đổi dữ liệu trước khi phân loại
- Có hai hướng:
- - Phân tách đầu vào bằng một ranh giới phi tuyến
- - Biến đổi hình 2D trên mặt phẳng thành một hình 3D để có thể tách.
### A. Từ Perceptron đến MLP
- Vượt qua bài toán XOR để phát triển mạng nơ ron nhân tạo :
- - chạy hai perceptron song song, rồi nối thêm perceptron thứ ba nối tiếp. Việc này cần một lớp ẩn.Lớp ẩn “bẻ cong” mặt phẳng toạ độ, biến không gian đặc trưng x thành không gian mới z mà ở đó dữ liệu tách được bằng đường thẳng.
### B. Ba ý tưởng cốt lõi của MLP
- Thêm lớp ẩn: biến đổi không gian đặc trưng thành không gian thuận lợi hơn cho phân loại
- Dùng hàm kích hoạt sigmoid cho phép quyết định linh hoạt hơn.
- Dùng thuật toán lan truyền ngược (backpropagation):điều chỉnh trọng số dựa trên sai số.
### C. Xây dựng lớp ẩn 
- B1:Biến đổi không gian đặc trưng :
- - Kết hợp song song hai Perceptron để tạo chiều rộng cho lớp ẩn
- B2:Liên kết cấu trúc( hoàn thiện MLP) :
- - Kết hợp Perceptron theo chuỗi: kết nối đầu ra của lớp ẩn nối tiếp vào perceptron lớp sau theo chuỗi
- B3:Kích hoạt khả năng tự học :
- - Thay thế hàm ngưỡng(cứng) bằng hàm Sigmod(mềm,trơn)
## 1.3. Từ MLP đến Deep Learning
- ANN (mạng nơ-ron nhân tạo) mô phỏng kết nối nơ-ron sinh học, có thể có nhiều lớp ẩn.
- Với bài toán cần biên phi tuyến thì dùng ANN có lớp ẩn.
- Lan truyền xuôi (forward propagation): ở mỗi lớp, nhân ma trận trọng số với vector đầu vào, cộng bias, rồi đưa qua hàm kích hoạt:
x¹ = Activation(W · x + b). Lặp lại qua từng lớp cho tới đầu ra.
## 1.4. Lan truyền ngược và Gradient Descent
- Lan truyền xuôi đến cuối, rồi lấy hiệu giữa giá trị dự đoán ŷ và giá trị thật y, đó là sai số (loss).
- Lan truyền ngược :
Chu trình huấn luyện: dự đoán → tính Loss → lan truyền gradient ngược → cập nhật trọng số → dự đoán lại. Mục tiêu là giảm hàm mất mát (Error/Loss).
## 1.5. Các hàm kích hoạt (Activation Function)
| Hàm | Đặc điểm | TensorFlow |
|-----| -------- | ---------- |
| Step | Dùng nhiều thời kỳ đầu; đầu ra 0/1 | _ |
| Sigmoid | Đầu ra trong (0, 1) | tf.math.sigmoid(x) |
| Tanh | Giống sigmoid nhưng đầu ra trong (−1, +1) | tf.math.tanh(x)|
| Linear | Đầu ra = đầu vào; dùng cho hồi quy | - |
| ReLU | Rectified Linear Unit, dạng đơn giản: max(0, x) | - |
| Leaky ReLU | Phiên bản “chỉnh nhẹ” của ReLU | - |
| Softmax | Tổng quát hóa sigmoid cho nhiều lớp; thường đặt ở lớp ra của bài toán phân loại | tf.nn.softmax(x) |
## 1.6. Deep Learning
- Deep Neural Network (DNN) có nhiều lớp ẩn; càng nhiều lớp ẩn thì mức trừu tượng hóa càng cao.
- Cấu trúc tổng quát: Input Layer → Hidden Layers → Output Layer; đầu vào X được chuyển thành đầu ra Y.
## 1.7. Các vấn đề của DNN và cách khắc phục
### a. Tín hiệu biến mất hoặc bùng nổ khi lan truyền qua các lớp
- Tránh dùng Sigmoid, dùng ReLU thay thế
- khởi tạo trọng số hợp lý để phương sai của các nút trong gần như không đổi: không khởi tạo hằng số (như 1); nên khởi tạo ngẫu nhiên quanh 0 với độ lệch chuẩn tỉ lệ nghịch với số nút của các lớp lân cận
### b. Overfitting
- Regularization trọng số: L1 hoặc L2
- Dropout: ngẫu nhiên loại bỏ một số nút khi huấn luyện để tránh phụ thuộc quá mức. Khi dự đoán thì dùng toàn bộ nút và cạnh.
- Data augmentation: thêm nhiễu, tạo dữ liệu mới bằng các phép biến đổi.
## 1.8.Giải bài tập 
### Bài 1.Hàm OR
- Trọng số: w₀ = −0,5; w₁ = 1; w₂ = 1
=> Bảng chân trị: (0,0)→F; (1,0)→T; (0,1)→T; (1,1)→T.
### Bài 2.Hàm AND
- Trọng số đúng: w₀ = −1,5; w₁ = 1; w₂ = 1
=>Bảng chân trị: chỉ (1,1) → T.
### Bài 3. Hàm XOR (cần MLP)
#### 3a. Chứng minh perceptron đơn không giải được

- XOR: (0,0)→F, (1,1)→F, (1,0)→T, (0,1)→T. Hai điểm “T” và hai điểm “F” nằm chéo nhau trên hình vuông nên không có đường thẳng nào tách được. Vì vậy chỉ đạt tối đa 3/4 = 75%.
#### 3b. Lời giải bằng 3 perceptron
- Dùng XOR(x₁,x₂) = AND( OR(x₁,x₂), NAND(x₁,x₂) ).
- - Perceptron ① (z₁ = OR): (w₀, w₁, w₂) = (−0,5; 1; 1)
- - Perceptron ② (z₂ = NAND): (w₀, w₁, w₂) = (1,5; −1; −1)
- - Perceptron ③ (AND trên z₁, z₂): (w₀, w₁, w₂) = (−1; 1; 1)
| x1 | x2 | s1 | z1 | s2 | z2 | s3=-1+z1+z2 | y |
|----|----|----|----|----|----|-------------|---|
| 0 | 0 | -0.5 | -1 | 1.5 | 1 | -1 | -1(F) |
| 1 | 0 | 0.5 | 1 | 0.5 | 1 | 1 | 1(T) |
| 0 | 1 | 0.5 | 1 | 0.5 | 1 | 1 | 1(T) |
| 1 | 1 | 1.5 | 1 | -0.5 | -1 | -1 | -1(F) |
-   Kết quả đúng XOR. Ở không gian mới (z₁, z₂) hai lớp tách được bằng một đường thẳng, nên đây chính là “bẻ cong mặt phẳng”.











