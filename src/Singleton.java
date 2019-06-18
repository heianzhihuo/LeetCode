
public class Singleton {
	private static volatile Singleton uniqueInstance;
	private Singleton() {
	}
	public static Singleton getUniqueInatance() {
		//先判断对象是否实例化过，如果没有实例化过才进入加锁代码
		if(uniqueInstance==null) {
			//类对象加锁
			synchronized (Singleton.class) {
				if(uniqueInstance==null) {
					uniqueInstance = new Singleton();
				}
			}
		}
		return uniqueInstance;
	}
}
