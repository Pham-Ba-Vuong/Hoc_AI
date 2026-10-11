# CƠ BẢN VỀ NEURAL NETWORK
## 1.1 Cấu trúc perceptron
- Gồm lớp đầu vào và lớp đầu ra, được coi là cấu trúc một lớp vì lớp đầu vào không tính toán.
- Nút bias x₀ luôn bằng 1, trọng số tương ứng là w₀.
- Lớp đầu ra có một nút: tính tổng s = w₀x₀ + w₁x₁ + … + w_d·x_d, rồi đưa qua hàm kích hoạt ngưỡng (step/threshold): s ≥ 0 cho 1, ngược lại cho −1.
- Perceptron phân loại đúng 100% tập huấn luyện
- Perceptron đơn chỉ đạt tối đa 75% độ chính xác với XOR.
## 1.2. Multilayer Perceptron (MLP)
### a. XOR cần hidden layer
- Có hai hướng: tách phi tuyến, hoặc nâng chiều (từ mặt phẳng 2D lên 3D).
- Cách giải: chạy hai perceptron song song, rồi nối thêm perceptron thứ ba nối tiếp. Việc này cần một lớp ẩn (hidden layer).
- Lớp ẩn “bẻ cong” mặt phẳng toạ độ, biến không gian đặc trưng x thành không gian mới z mà ở đó dữ liệu tách được bằng đường thẳng.
### b. Ba ý tưởng cốt lõi của MLP
- Thêm lớp ẩn: biến đổi không gian đặc trưng thành không gian thuận lợi hơn cho phân loại
- Dùng hàm kích hoạt sigmoid cho phép quyết định linh hoạt hơn.
- Dùng thuật toán lan truyền ngược (backpropagation): duyệt ngược từ lớp ra về lớp vào, tính gradient và cập nhật trọng số từng lớp.
## 1.3. Từ MLP đến Deep Learning
- ANN (mạng nơ-ron nhân tạo) mô phỏng kết nối nơ-ron sinh học, có thể có nhiều lớp ẩn.
- Với bài toán cần biên phi tuyến thì dùng ANN có lớp ẩn.
- Lan truyền xuôi (forward propagation): ở mỗi lớp, nhân ma trận trọng số với vector đầu vào, cộng bias, rồi đưa qua hàm kích hoạt:
x¹ = Activation(W · x + b). Lặp lại qua từng lớp cho tới đầu ra.
## 1.4. Lan truyền ngược và Gradient Descent
- Lan truyền xuôi đến cuối, rồi lấy hiệu giữa giá trị dự đoán ŷ và giá trị thật y, đó là sai số (loss).
- Lan truyền ngược: truyền sai số ngược về các lớp trước, cập nhật tham số bằng gradient descent, rồi lặp lại từ bước lan truyền xuôi.
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
## 1.7. Các vấn đề của DNN và cách khắc phục
### a. Tín hiệu biến mất hoặc bùng nổ khi lan truyền qua các lớp
- Tránh dùng Sigmoid, dùng ReLU thay thế
- hởi tạo trọng số hợp lý để phương sai của các nút trong gần như không đổi: không khởi tạo hằng số (như 1); nên khởi tạo ngẫu nhiên quanh 0 với độ lệch chuẩn tỉ lệ nghịch với số nút của các lớp lân cận
### b. Overfitting
- Regularization trọng số: L1 hoặc L2
- Dropout: ngẫu nhiên loại bỏ một số nút khi huấn luyện để tránh phụ thuộc quá mức. Khi dự đoán thì dùng toàn bộ nút và cạnh.
- Data augmentation: thêm nhiễu, tạo dữ liệu mới bằng các phép biến đổi.