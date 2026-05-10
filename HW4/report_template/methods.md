# Topic 3 - Mô tả 4 thành phần GA

## Fitness Function

Bài toán được tối ưu theo hướng ưu tiên tính khả thi trước, sau đó mới tối ưu chất lượng mềm của thời khóa biểu.

Ta dùng hàm mục tiêu dạng penalty và chuyển sang fitness tối đa:

- total_penalty = hard_penalty_weight * hard_violations + soft_penalty_weight * soft_penalty
- fitness = 1 / (1 + total_penalty)

Trong đó:

- hard_penalty_weight = 1000 và soft_penalty_weight = 1 để bảo đảm mọi vi phạm hard constraints bị phạt nặng hơn rất nhiều so với cải thiện mềm.
- hard_violations gồm toàn bộ ràng buộc cứng:
  - mỗi offering có đúng 2 buổi/tuần, không cùng ngày, không ngày liền kề
  - phòng phải đủ sức chứa theo class_registration_size
  - professor không dạy trùng giờ
  - chỉ dùng phòng khả dụng
  - cùng section không học 2 phòng tại cùng thời điểm
  - cùng phòng không chứa 2 section tại cùng thời điểm
  - professor không vượt quá 3 courses (được kiểm soát từ data generation và vẫn theo dõi như một hard check)
- soft_penalty dùng để nâng chất lượng lịch khi đã khả thi:
  - độ lệch phân bố tải theo timeslot (giảm dồn cục)
  - độ lệch sử dụng phòng (giảm lệch tải phòng)
  - idle gaps của section trong cùng ngày

Thiết kế này giúp GA hội tụ nhanh về nghiệm hợp lệ, đồng thời vẫn tiếp tục cải thiện độ mượt của lịch.

## Selection

Nhóm sử dụng Tournament Selection với kích thước giải đấu k = 4.

Quy trình:

1. Chọn ngẫu nhiên k cá thể trong quần thể.
2. Cá thể có total_penalty thấp nhất thắng giải và được chọn làm parent.
3. Lặp lại để lấy đủ parent cho bước lai ghép.

Lý do chọn:

- đơn giản, ổn định, không cần chuẩn hóa xác suất như roulette wheel
- giữ được áp lực chọn lọc vừa phải, ít nhạy với chênh lệch fitness cực lớn do hard penalty
- dễ kết hợp với elitism

## Crossover

Nhóm dùng Offering-based Uniform Crossover.

Với mỗi gene (mỗi gene tương ứng một offering gồm 2 session), child sẽ nhận toàn bộ gene từ parent A hoặc parent B theo xác suất 0.5. Cách lai này giữ nguyên cấu trúc theo offering, tránh phá vỡ cục bộ 2 session của cùng lớp.

Sau crossover luôn chạy repair để khôi phục tính hợp lệ cơ bản:

- sửa cặp ngày để luôn thỏa điều kiện không cùng ngày và không liền kề
- ép chỉ số day/slot vào miền hợp lệ
- thay room không hợp lệ bằng room hợp lệ gần nhất theo class size

Nhờ đó offspring vẫn đúng cấu trúc chromosome và giảm xác suất sinh cá thể quá xấu.

## Mutation

Mutation được thiết kế đa dạng nhưng có kiểm soát:

- room mutation: đổi phòng của một session theo tập phòng phù hợp kích thước lớp
- slot mutation: đổi timeslot của một session
- day-pair mutation: đổi đồng thời cặp ngày của 2 session trong cùng offering
- swap mutation: hoán đổi một session giữa hai offering ngẫu nhiên

Mỗi gene có xác suất mutation_rate để đột biến (mặc định 0.2). Sau mutation tiếp tục chạy repair giống sau crossover để bảo toàn cấu trúc và miền giá trị.

Cơ chế này giúp cân bằng exploration và exploitation:

- exploration nhờ thay đổi room/slot/day và swap
- exploitation nhờ repair + elitism giữ lại các cá thể tốt nhất qua mỗi thế hệ
