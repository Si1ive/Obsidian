注释

请求映射注释

@**mapping 如@postmapping，最基本的是@RequestMapping，在其参数中设置method=post 一样的效果

类和方法上都可以加

@PathVariable用来获取动态路径上的值，mapping后用/{**}来写路径

@RequestParam 获取参数,可以设置初始值，初始类型 required = false即 空参不报错 true 空参报错 直接返回400

如果参数只有一个name=”“ ，这个参数可以省略，多个参数就不能省略了

@RestController 就是@Controller和@ResponseBody组合而成的

@ResponseBody是将返回的对象转化为json格式，如果不转，前端是解析不了的，前端为什么直接就404了？

并且内容是填写在body里的，不是头

@ComponentScan控制去哪扫描要注入的类，默认是扫描当前包，所以如果把要注入的类放到当前包下，即便不打component他也能扫描注入？

@bean 是写在方法上的

@ConditionalOnClass注解 就是根据不同的情况来装载bean

classpath目录指的是resources文件夹

@AllArgsConstructor全参构造

@SpringBootApplication 其中包含三个注解

1）@EnableAutoConfiguration自动配置类 ,会自动将AutoConfigurationImportSelector类导入，这个类会检查符合要求的Configuration类，然后将其导入

2）SpringBootConfiguration 打开会发现啥都没有，他就等捅于Configuration

3）ComponentScan是扫描符合条件的Bean

xml是如何获取到类的 就是class方法吗

这个<>里面的名字是怎么定义的，那我能不能规定一套xml，来调用我写好的类呢？

xml配置又是如何转换到注释来完成的呢？

1）注释是通过反射获取到类，然后再判断类上的注释要进行什么操作，然后操作

xml是通过

为什么自定义appender和filter中都要设定成员变量并且设定set函数呢？这与xml之间是什么联系

springboot启动流程

springboot.run之后有四步

1）服务构建：通过SpringApplication来加载各种启动器

如：资源加载器，主方法类，web服务器（servlet），以及上下文初始化器，监听器等等

2）环境准备：听不懂

3）容器创建:

applicationContext,会创建用来管理bean的beanfactory，

以及用来解析Component，componentScan等注解的配置类后处理器ConfigurationClassPostProcessor

用来解析Autowired，value，inject等注解的 AutowiredAnnotationBeanPostProcessor

可以直接统称为BeanPostProcessor吗