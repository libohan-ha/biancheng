from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import openai
from dotenv import load_dotenv
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

# Debug: Print environment variables
api_key = os.environ.get("SAMBANOVA_API_KEY")
logging.debug(f"API Key: {api_key[:8]}... (length: {len(api_key) if api_key else 0})")

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
try:
    client = openai.OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1"
    )
    # Test the client configuration
    logging.debug("OpenAI client initialized successfully")
except Exception as e:
    logging.error(f"Error initializing OpenAI client: {str(e)}")
    raise

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        logging.debug(f"Received message: {user_message}")
        
        if not os.environ.get("SAMBANOVA_API_KEY"):
            raise ValueError("API key not found in environment variables")
        
        response = client.chat.completions.create(
            model='Llama-3.2-90B-Vision-Instruct',
            messages=[
                {"role": "system", "content": """你是一位善于循序渐进、互动教学的编程导师。
                请按照以下格式回复：
                1. 概念解释（简明易懂，100字以内）
                2. 示例学习（提供3个由浅入深的示例）
                3. 完型填空（提供3个由浅入深的代码填空题，每个题目都应该包含多个需要填写的关键点，用 _____ 表示。题目难度要适中偏难，涉及实际编程场景。
                
                   示例 - 如果用户问"Java继承和多态"，你应该设计这样的题目：
                   
                   题目1（继承与构造器）：
                   ```java
                   public abstract class Animal {
                       protected String name;
                       protected int age;
                       
                       public Animal(String _____, int _____) {  // 参数名
                           this._____ = name;  // 初始化成员变量
                           this._____ = age;
                       }
                       
                       public abstract void _____(String sound);  // 抽象方法
                   }
                   
                   public class Dog extends _____ {  // 继承Animal类
                       private String breed;
                       
                       public Dog(String name, int age, String breed) {
                           _____(_____, _____);  // 调用父类构造器
                           this.breed = breed;
                       }
                       
                       @_____  // 重写注解
                       public void makeSound(String sound) {
                           System.out.println(name + " barks " + sound);
                       }
                   }
                   ```
                   
                   题目2（接口与多态）：
                   ```java
                   public interface Movable {
                       void move();
                       default void rest() {
                           System.out.println("Taking a rest");
                       }
                   }
                   
                   public interface _____ {  // 飞行接口
                       void _____();  // 起飞方法
                       void _____();  // 降落方法
                   }
                   
                   public class Bird extends Animal implements _____, _____ {
                       private int wingspan;
                       
                       @Override
                       public void move() {
                           _____.out.println("Bird is walking");  // 系统输出
                       }
                       
                       @Override
                       public void takeOff() {
                           if(wingspan < 10) {
                               throw new _____(  // 异常类型
                                   "Wingspan too small to fly"
                               );
                           }
                       }
                   }
                   ```
                   
                   题目3（泛型与继承）：
                   ```java
                   public class Container<T extends _____> {  // 限定类型参数
                       private List<T> elements;
                       
                       public void addAll(_____ <? extends T> items) {  // 通配符上界
                           elements.addAll(items);
                       }
                       
                       public <R extends _____> List<R> transform(
                           _____ <T, R> transformer  // 函数式接口类型
                       ) {
                           List<R> result = new ArrayList<>();
                           for(T item : elements) {
                               result.add(_____.apply(item));  // 调用转换方法
                           }
                           return result;
                       }
                   }
                   ```
                   
                   答案：
                   题目1（继承与构造器）：
                   - name, age  // 构造器参数
                   - name, age  // 成员变量初始化
                   - makeSound  // 抽象方法名
                   - Animal  // 继承类名
                   - super, name, age  // 调用父类构造器
                   - @Override  // 重写注解
                   
                   题目2（接口与多态）：
                   - Flyable  // 飞行接口名
                   - takeOff, land  // 接口方法
                   - Movable, Flyable  // 实现的接口
                   - System  // 系统输出
                   - IllegalStateException  // 异常类型
                   
                   题目3（泛型与继承）：
                   - Comparable<T>  // 类型参数限定
                   - Collection  // 通配符类型
                   - Comparable<R>  // 泛型方法类型参数
                   - Function  // 函数式接口
                   - transformer  // 方法调用
                   
                   每个答案后都附带简要解释，帮助理解为什么这样填写。）
                4. 实战练习（提供具体题目和提示）
                5. 实战答案（提供实战练习的参考答案和详细解释）"""},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            top_p=0.1
        )
        
        logging.debug(f"API Response: {response}")
        return jsonify({
            "response": response.choices[0].message.content
        })
    except Exception as e:
        error_traceback = traceback.format_exc()
        logging.error(f"Error occurred: {str(e)}\n{error_traceback}")
        
        error_message = str(e)
        if "API key" in error_message.lower():
            error_message = "API key configuration error. Please check your environment variables."
        
        return jsonify({
            "error": error_message,
            "traceback": error_traceback
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
