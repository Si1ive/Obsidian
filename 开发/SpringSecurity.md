跨域请求伪造，窃取名为JSESSIONID的Cookie信息，模仿登录，但是现在的浏览器都会判断其他信息，不仅仅看这一个sessionid了

会话固定攻击，恶意修改JSESSIONID的Cookie信息，你会顶着这个id去登陆，也就是它不需要窃取你的id了。tomcatHttpOnly会保护cookie信息不能修改

## 在mvc中使用

1. 需要先编写初始化器，就是一个继承了抽象的初始化类，也不用实现什么东西，底层是通过多个过滤器来实现的

（但是在springboot项目中没有看到初始化类，是被自动初始化了吗，在哪里体现）

2. 然后再编写配置类，配置类在springboot中仍有体现，因为要设置东西，所以保留？并且mvc中还需要在main启动器那里启动配置类

配置类中主要是通过filterChain方法来实现对HttpSecurity的配置，http中有大量的方法用来设置，每个方法的参数都是一个函数式接口，不过是接口通过泛型对参数进行了区分，从而实现了http不同方法要填入的不同的表达式的效果。

3. 配置方法总结

formLogin 设置表单登录

sessionmanager 就是对session进行设置，前后端分离后session就不用了，把sessionpolicy的策略设置成stateless

为什么有一版本需要.and再重新返回httpsecurity，有一版本不需要呢？

.authorizeHttpRequests对请求认证的设置

后面接matchers匹配路径，再对路径进行设置，如anonymous即可以匿名访问 甚至你带token反而访问不了，也就是登录页面只有匿名才能看见

.permitAll()是真的无论登不登陆都可以访问

anyRequest()指其他的任意请求

authenticated()是指登陆过可以访问

4. 过滤器总结

如usernamepasswordautheractionfilter，

5. jwt与过滤器

获取token后用jwt处理，用jwt处理出的信息，在redis中把用户信息读出来，再用usernamepasswordautheractionfilter处理，再存入SecurityContextHolder

(第一次登录时，要把信息存到redis中)

![0](https://note.youdao.com/yws/res/270/WEBRESOURCE036ba7e2b3441ab1b66980b359a39ccc)

但是在项目中没找到providermanager 被什么代替了？

并且没有自定义存储数据库，是因为security自己整合了存储数据库的操作？

并且因为没有将用户登录信息往redis中存储，所以也没有redis的序列化器配置，因为redis在这个项目中仅存储了验证码？

![0](https://note.youdao.com/yws/res/279/WEBRESOURCE7b6c6273b71da4e876d54a02343f0296)

6. 权限

![0](https://note.youdao.com/yws/res/360/WEBRESOURCE599d5e982d5253f94b71eb23cd8cebf1)

7. 授权

也是分注释和xml两种，但xml一般用于静态资源，所以前后端分离中不再使用，而使用注释

在配置类上启动enablegloblmethod这个废弃了，那换成什么了@EnableMethodSecurity

@PreAuthorize("hasAuthority('demo')") 执行方法时，检验这个方法是否有这个权限 这个用来检查权限信息

在jwtconfig中设置权限信息，（为什么不在securityconfig当中设置权限信息呢？

UsernamePasswordAuthenticationToken先通过用户信息获得一个权限token，再通过SecurityContextHolder放到SecurityContext中

## JWT

jwt的生成方式，怎么这么多种

1.用JWTS.builder创建再转成string

2.直接jwt create 生成string