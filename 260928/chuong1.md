# A.Tìm hiểu
## 1.NHẬP MÔN TRÍ TUỆ NHÂN TẠO
### 1.KHÁI NIỆM VỀ AI
#### 1.1 Định nghĩa AI
- AI là khả năng của chương trình máy tính hoặc máy móc trong việc suy nghĩ và học.
- Ba quá trình cốt lõi:
- - Học: thu nhận thông tin và các quy tắc sử dụng thông tin.
- - Suy luận: dùng quy tắc để đi đến kết luận gần đúng hoặc chắc chắn.
- - Tự hiệu chỉnh.
- Dữ liệu ngày nay rất dồi dào, cả có cấu trúc lẫn phi cấu trúc. Machine Learning ra đời cuối thế kỷ 20 như nhánh con của AI, gồm các thuật toán tự học để trích xuất và dự đoán tri thức từ dữ liệu.
- ML là cải thiện dần mô hình dự đoán và chất lượng ra quyết định bằng cách trích xuất tri thức hiệu quả hơn.
#### 1.2 Các loại và tập con của AI
- ANI (Weak AI)
- - Khả năng: Làm một chức năng cụ thể mà con người làm được
- - Ví dụ: Loa thông minh, xe tự lái
- - Tiến độ: Phần lớn tiến bộ gần đây nằm ở đây
- AGI (Strong AI)
- - Khả năng: Làm được mọi việc con người làm được
- - Ví dụ: Chưa đạt được
- - Tiến độ :Chậm hơn ANI
- Cả khóa học tập trung vào ANI.

Quan hệ lồng nhau: AI ⊃ ML ⊃ DL
- - AI: mọi kỹ thuật giúp máy tính bắt chước hành vi con người.
- - ML: tập con của AI, dùng phương pháp thống kê để máy cải thiện theo kinh nghiệm.
- - DL: tập con của ML, giúp tính toán mạng nơ-ron nhiều lớp trở nên khả thi.
####  1.3. Định nghĩa Machine Learning
- ML là một lĩnh vực của AL , nghiên cứu các thuật toán máy tính tự động cải thiện thông qua ví dụ và kinh nghiệm
#### 1.4Các ngành liên quan đến ML
- ML là lĩnh vực liên ngành: xác suất thống kê, khoa học máy tính, lý thuyết cơ sở dữ liệu, khoa học nhận thức, thần kinh học, nhận dạng mẫu, data mining, data science.
- - ML và Thống kê
- - - Thống kê nhấn mạnh suy diễn và kiểm định
- - - ML giải các bài toán khó thiết kế hoặc khó lập trình thuật toán.
- - ML và Data Mining
- - - Data mining : khám phá có hệ thống và tự động các mẫu có ý nghĩa trong dữ liệu lớn. ML: chương trình học và dự đoán, cộng với việc nghiên cứu, xây dựng thuật toán cho quá trình đó.
- - Thống kê truyền thống và Data mining
- - - Thống kê truyền thống :
- - - - Giả định :	Có giả định về phân phối/mô hình của tổng thể
- - - - Cách làm :	Suy diễn tham số tổng thể từ mẫu
- - - - Dữ liệu	Không đòi hỏi lớn
- - - Data mining
- - - - Giả định :
Không cần giả định
- - - - Cách làm:
Dùng toàn bộ dữ liệu tổng thể
- - - - Dữ liệu:
Bắt buộc dữ liệu lớn
#### 1.5. Các loại phân tích dữ liệu bằng ML và DL
- ML: dữ liệu huấn luyện → trích xuất đặc trưng (do con người thiết kế) → mô hình → phân loại
- DL: dùng mạng nơ-ron nhân tạo nhiều lớp, tự học đặc trưng.
- Bốn bài toán của Data Mining
- - 1.Prediction: xây mô hình từ dữ liệu có sẵn rồi dự đoán ca mới. Ví dụ: dự đoán chất lượng thành phẩm từ nguyên liệu và môi trường.
- - 2.Classification: xác định ca thuộc nhóm nào trong các nhóm có sẵn. Ví dụ: xếp hạng tốt/bình thường/xấu.
- - 3.Clustering: gom các đối tượng có thuộc tính tương tự thành cụm. Ví dụ: gom các quy trình có đặc tính giống nhau.
- - ssociation Rule: tìm quan hệ "mẫu này xuất hiện kéo theo mẫu kia". Ví dụ: dự đoán ảnh hưởng toàn quy trình khi một công đoạn bất thường.
- a.Học có giám sát
- - Dữ liệu huấn luyện có nhãn (đáp án mong muốn), ví dụ tập phân loại spam.
- - Biểu diễn quan hệ giữa biến giải thích (feature) và biến mục tiêu (target) và dự đoán quan sát tương lai. Hợp với nhận dạng, phân loại, chẩn đoán, dự đoán.
- - Biến mục tiêu định tính → Thuật toánClassification. Biến mục tiêu định lượng → Regression.
- - Thuật toán
- - - Classification: K-NN, Logistic Regression, ANN, Decision Tree, SVM, Naïve Bayes, Ensemble (Random Forest…).
- - - Regression: Linear, hồi quy mở rộng (Polynomial, Nonlinear, Penalized), ANN, Decision Tree, SVM Regression, PLS, Ensemble
- - Slide dùng sơ đồ scikit-learn cheat-sheet để chọn thuật toán.
- - Thiếu dữ liệu: dữ liệu ít thì phải dùng thống kê. Tối thiểu khoảng 30 mẫu để ước lượng tham số tổng thể, trên 30 có thể giả định phân phối chuẩn. Data mining/ML cần dữ liệu rất lớn, tiêu chuẩn khoảng ≥ 100.000 mẫu.
- b.tự học
- - Không có nhãn, hệ thống tự học. Dùng cho mô tả, rút đặc trưng, tìm mẫu. Tính data mining mạnh hơn.
- - Nhóm thuật toán:
- - - Phân cụm: K-Means, DBSCAN, Hierarchical (HCA), phát hiện bất thường/outlier, One-Class SVM, Isolation Forest.
- - - Trực quan hóa và giảm chiều: PCA, Kernel PCA, Local Linear Embedding, t-SNE.
- - - Luật kết hợp: Apriori, Eclat.
- c.Batch và Online Learning
- - Batch: hệ thống không học dần được, phải huấn luyện lại toàn bộ.
- -  Online: đưa dữ liệu lần lượt từng mẫu hoặc từng mini-batch. Phù hợp dữ liệu rất lớn.
- d.Instance-based và Model-based
- - Instance-based: ghi nhớ mẫu huấn luyện, khái quát hóa bằng cách so sánh mẫu mới với mẫu đã học qua độ đo tương tự.
- - Model-based: xây mô hình từ mẫu rồi dùng mô hình để dự đoán.
#### 1.6.Quy trình 6 bước
- 1.Hiểu nghiệp vụ và xác định vấn đề:
- 2.Thu thập dữ liệu
- - Trích từ kho nội bộ (data warehouse, data mart) bằng SQL, hoặc từ nền tảng big data (Hadoop).
- - Dữ liệu ngoài lấy bằng web scraping hoặc API.
- - Nguồn: nội bộ (thủ công, log collector), bên ngoài (web crawling, cảm biến, media).
- 3.Tiền xử lý và khám phá:Các kỹ thuật:
- - Chuẩn hóa: Z-transform, Normalization.
- - Về phân phối chuẩn: Log transform (dữ liệu phân phối nghịch đảo), Square root transform
- - Phân loại hóa: Discretization (chia biến liên tục thành khoảng), Binarization (biến giả 0/1).
- - Lấy mẫu: Random, Systematic, Stratified, Cluster, Multistage.
- - Giảm chiều: Factor Analysis, PCA.
- - Nén tín hiệu: Fourier, Wavelet.
- 4.Huấn luyện mô hình:
- - Có giám sát: chia dữ liệu huấn luyện và kiểm định/đánh giá, hoặc dùng cross-validation.
- - Không giám sát: không có giá trị mục tiêu nên chủ yếu là rút mẫu qua phân tích.
- 5.Đánh giá hiệu năng
- - Mô hình thường thiên lệch trên dữ liệu đã huấn luyện nên phải dùng tập đánh giá (ví dụ confusion matrix, precision, recall).
- - Không giám sát thường không có tập đánh giá, nên đánh giá qua khả năng diễn giải các quy tắc rút ra.
- 6.Cải thiện mô hình và ứng dụng
- - Hiếm khi giải xong một lần. Liên tục đổi tham số, phương pháp ước lượng, thử thuật toán khác.
- - Không có tiêu chí tuyệt đối cho việc "đủ tốt", tùy bài toán và lĩnh vực.
- - Khi đạt yêu cầu thì áp dụng vào nghiệp vụ, đôi khi cần thêm việc tự động hóa hoặc liên kết hệ thống.
#### 1.7.Vì sao dùng ML
- Lập trình luật truyền thống: luật ngày càng dài, phức tạp, khó bảo trì. Bộ lọc spam ML thì tự học từ/cụm từ nào là dấu hiệu spam.
- Chu trình truyền thống: nghiên cứu → viết luật → đánh giá → phân tích lỗi → lặp. Chu trình ML: nghiên cứu → huấn luyện thuật toán bằng dữ liệu → đánh giá → phân tích lỗi.
- Điểm mạnh:
- - Bài toán cần nhiều chỉnh tay: một mô hình ML đơn giản hóa code, tăng hiệu năng.
- - Bài toán phức tạp chưa có lời giải truyền thống (nhận dạng giọng nói): ML tìm được lời giải.
- - Môi trường biến động: hệ thống thích nghi với dữ liệu mới.
- - Thu được insight từ dữ liệu lớn.
#### 1.8. Hạn chế của ML
- 1.Thiếu dữ liệu huấn luyện
- 2.Dữ liệu không đại diện:
- - Sampling noise: mẫu nhỏ không đại diện do ngẫu nhiên.
- - Sampling bias: mẫu lớn nhưng phương pháp lấy mẫu sai nên vẫn không đại diện.
- 3.Dữ liệu chất lượng kém: lỗi, outlier, nhiễu làm khó tìm mẫu. Nếu rõ ràng là outlier thì bỏ hoặc sửa.
- 4.Đặc trưng không liên quan: thành công phụ thuộc vào feature engineering.
- - Feature selection: chọn đặc trưng hữu ích nhất.
- - Feature extraction: kết hợp đặc trưng để giảm chiều.
- 5.Overfitting: khớp quá mức dữ liệu huấn luyện. Khắc phục bằng regularization (ràng buộc để đơn giản hóa mô hình).
- 6.Underfitting: mô hình quá đơn giản. Khắc phục bằng mô hình mạnh hơn (nhiều tham số hơn), đặc trưng tốt hơn, giảm ràng buộc regularization.
### 2.ỨNG DỤNG CỦA AI
#### 2.1Tổng quan
- Nhờ ML có bộ lọc spam, nhận dạng chữ và giọng nói, tìm kiếm, xe tự lái. Y tế cũng tiến bộ lớn: DL chẩn đoán ung thư da gần bằng con người (Esteva, Nature 2017), và dự đoán cấu trúc 3D của protein,
- Ứng dụng theo các loại bài toán:Image Classification;	
Semantic Segmentation;	
Text Classification	;
Text Summary;	
Language Understanding;	
Regression;
Voice Recognition;
Outlier Detection;
Clustering;Data Visualization;	
Recommendation;
Reinforcement Learning
#### 2.2Nhận dạng hình ảnh
- Nhận diện địa điểm, logo, người, vật thể, tòa nhà… Computer vision còn gồm phát hiện sự kiện, tái dựng ảnh, theo dõi video.
- Các nhiệm vụ: Semantic Segmentation (gán nhãn theo pixel), Classification + Localization (một vật thể), Object Detection (nhiều vật thể), Instance Segmentation.
#### 2.3. Computer Vision và Machine Vision
- Computer Vision: lĩnh vực liên ngành giúp máy hiểu ảnh/video. Ứng dụng: nông nghiệp, địa chất, sinh trắc học, AR, ảnh y tế, robot, kiểm tra công nghiệp, an ninh.
- Machine Vision: thiên về kỹ thuật hệ thống, tích hợp công nghệ để kiểm tra tự động, điều khiển quy trình, dẫn hướng robot, thường trong công nghiệp.
- Case study 
- - Acquire Automation: kiểm tra chai 360° (nắp, seal, vị trí, nhãn, màu, mã vạch) và cung cấp thống kê sản xuất thời gian thực.
- - Cognex VisionPro ViDi: phần mềm DL cho nhà máy (phát hiện lỗi, phân loại vật liệu, kiểm tra lắp ráp, đọc chữ). Chỉ cần vài trăm ảnh, huấn luyện nhanh, chi phí thấp, hướng tới người không chuyên.
- - Focal: camera nhỏ chụp mỗi 30 phút để phát hiện hàng hết. Nhân viên thủ công mất khoảng 4 giờ/ngày. Thanh toán bằng camera trên băng chuyền giảm thời gian giao dịch tới 60%.
#### 2.4.Giọng nói và ngôn ngữ
- NLP: giao thoa ngôn ngữ học, CNTT và AI, về tương tác giữa máy tính và ngôn ngữ tự nhiên. Thách thức: nhận dạng giọng nói, hiểu (NLU), sinh (NLG). Ứng dụng: dịch máy, truy xuất thông tin, hỏi đáp, trích xuất, tóm tắt, phân loại văn bản.
- Speech Recognition: nhận biết nội dung lời nói.
- Voice Recognition: nhận biết giọng, cao độ, ngữ điệu của người nói bất kể ngôn ngữ (xác minh và nhận dạng người nói). Quy trình: âm thanh analog → chuyển sang số → nhận dạng mẫu.
- Case study:
- - Hello Barbie: micro ghi âm → gửi server → AI chọn câu trả lời → phát ra loa. Nhớ điều trẻ nói, dự đoán hội thoại của trẻ 3–9 tuổi, trao đổi tới ~200 lượt.
- - Personetics Assist: chatbot tài chính phục vụ 24/7 không tốn nhân công. Tích hợp dữ liệu giao dịch cá nhân, dùng phân tích dự đoán để tư vấn trước một bước. Thay được tác vụ chuyển tiền, đặt chỗ, đổi mật khẩu.
### 3.CÁC KỸ THUẬT AI
#### 3.1 Edge AI
- Chạy AI trực tiếp trên thiết bị biên (điện thoại, xe, thiết bị đeo) thay vì gửi lên cloud, để xử lý cục bộ và phản hồi nhanh.
-Xe tự lái phải phản ứng thời gian thực và hoạt động cả khi không có internet, vì độ trễ có thể gây tai nạn.
- Thang tính toán: Cloud → Fog → Edge.
-  Use case: camera thông minh trong nhà, nhận diện khuôn mặt/vật thể trên thiết bị (dữ liệu người dùng không rời thiết bị), quyết định lái tức thời, drone, robot, camera trông trẻ.
#### 3.2.Ảnh y tế và chẩn đoán
- phần mềm AI IDx-DR sàng lọc bệnh võng mạc tiểu đường
- thiết bị soi đáy mắt cầm tay và thuật toán hỗ trợ chẩn đoán cho người thiếu điều kiện y tế.
#### 3.3.Xe tự lái
- 5 cấp tự động (SAE):
- - L1: hỗ trợ lái ("Feet Off").
- - L2: tự động một phần ("Hands Off").
- - L3: tự động có điều kiện ("Eyes Off").
- - L4: tự động cao ("Mind Off").
- - L5: hoàn toàn (robo-taxi, mọi điều kiện).
- - L1–2 thuộc ADAS.
- Disengagement (người lái phải tiếp quản)
- Trung Quốc: Baidu Apollo (4/2017) là nền tảng mở, quy tụ hãng xe, Tier 1, chip, bản đồ, LiDAR, HĐH, cloud.
- "Unbundling" xe tự lái: hệ thống lái tự động (Drive.ai, Momenta, Pony.ai), thị giác máy tính (DeepScale, Prophesee), dữ liệu và mô phỏng (Cognata, NVIDIA); cảm biến LiDAR, camera, radar, định vị, bản đồ, V2X.
#### 3.4. Học tăng cường
- Agent thực hiện action trong môi trường, nhận state và reward, học để tối đa hóa phần thưởng. Kết hợp mạng sâu là Deep RL (ví dụ AlphaGo).
- Ứng dụng: game (Go, poker, Dota, StarCraft), robot, NLP, y tế, giáo dục, giao thông (đèn tín hiệu thích ứng), năng lượng, tài chính (tối ưu danh mục), thương mại điện tử, nghệ thuật.
#### 3.5.AI hội thoại
- Chuỗi trợ lý giọng nói (Bixby): ASR (giọng nói → văn bản) → NLU (hiểu ý) → Dialog Management → NLG (sinh câu trả lời) → phản hồi. Nền tảng là ML và mạng nơ-ron sâu.
#### 3.6. GAN, XAI, dữ liệu tổng hợp
- GAN: tạo ảnh/video giả rất chân thực.
- XAI (DARPA):giải thích được lý do và cho người dùng biết khi nào nên tin, khi nào hệ thống sẽ sai, cách sửa lỗi.
- Dữ liệu tổng hợp: GAN học từ MRI công khai để tạo ảnh MRI khối u não giả. Lợi ích: tăng độ chính xác phân đoạn khối u và ẩn danh hóa, giảm rủi ro dùng dữ liệu bệnh nhân khi chưa được phép.
### 4. XU HƯỚNG VÀ THỊ TRƯỜNG AI
#### 4.1.Xu hướng
- Hai trục: độ mạnh thị trường và mức áp dụng trong ngành. Bốn vùng: Experimental, Transitory, Necessary, Threatening.
- Xu hướng nổi bật: nhận dạng khuôn mặt, ảnh y tế, edge computing, conversational agents, synthetic training data, khám phá thuốc, cửa hàng không thanh toán, học tăng cường, Explainable AI, GAN, federated learning, capsule networks…
#### 4.2.Năng lượng bền vững
- Green AI: gắn sức mạnh tính toán với phát thải carbon để quản lý chi phí carbon của AI.
#### 4.3.Dịch vụ tài chính
- AI + sinh trắc học giúp xác minh danh tính nhanh, chính xác hơn, và tội phạm không thể truy cập chỉ bằng thông tin đăng nhập.
#### 4.4 Chính phủ
- Wisconsin DWD xử lý tồn đọng hồ sơ thất nghiệp bằng AI trên cloud
- Dùng Google Cloud DocAI (trích xuất dữ liệu từ tài liệu) và Human-in-the-Loop (con người xem xét để tăng độ chính xác).
#### 4.5.Y tế
- FDA công bố kế hoạch cho phần mềm AI/ML như thiết bị y tế (SaMD).
#### 4.6.IoT và AI trong nông nghiệp
- Plenty (San Francisco) trồng cây theo chiều dọc trong nhà: đèn LED thay ánh nắng, robot chăm sóc, AI quản lý nước, nhiệt độ, ánh sáng và tự học tối ưu.




