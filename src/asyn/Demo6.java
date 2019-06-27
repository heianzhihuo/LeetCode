package asyn;

public class Demo6 extends BaseDemo {
	
	private int s = 0;//信号量的简单使用
	
	@Override
	public void callback(long response) {
		System.out.println("得到结果");
		System.out.println(response);
		System.out.println("调用结束");
		synchronized (this) {
			s++;
			notifyAll();
		}
	}
	
	public static void main(String[] args) {
		
		Demo6 demo6 = new Demo6();
		
		demo6.call();
		//这里用简单的wait和notify实现了异步调用
		//这里简单的使用了
		
		try {
			Thread.sleep(5*1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		try {
			//这里应当是使用的是对象锁，而不是类锁
			//因为当执行到这里的时候，已经有两个线程同时在使用这个类，如果这里使用的是类锁，则这里会报错
			synchronized (demo6) {
				while(demo6.s==0) demo6.wait();
			}
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		System.out.println("主线程内容");
	}
}
