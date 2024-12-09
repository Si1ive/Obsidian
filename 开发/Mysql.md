# Mysql
## 字段类型

### char与varchar

char 是固定长度，创建时就要确定长度，varchar是变长，varchar（M）中M是指最大长度，并且varchar存储时会额外存储字段长度，所以会多占一些

所以char 适合长度固定或类似的，varchar来存储名字这种不定长的

### Decimal与Float

前者是十进制精确的小数，后者是浮点数，小数部分不能精确表示

### 时间数据类型

DateTime表示的范围大，占用字节多，不具备时区信息

TimeStamp表示范围小，占用字节少，具备时区信息，切换时区时会跟着变，但变得过程有性能消耗

按数值存储时间戳 不能使用时间数据类型的api，难以阅读
# Mybatis
@TableName("db_account") 指定实体类对应的表名

@TableId(type = IdType.AUTO) 指定主键

配置类中的分页插件是什么作用