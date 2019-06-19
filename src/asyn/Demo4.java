package asyn;

import java.util.concurrent.CountDownLatch;

public class Demo4 extends BaseDemo {

	private final CountDownLatch CountDownLatch = new CountDownLatch(1);
	
	@Override
	public void callback(long response) {
		System.out.println("得到结果");
		System.out.println(response);
		System.out.println("调用结束");
		CountDownLatch.countDown();
	}

	public static void main(String[] args) {
		
		Demo4 demo4 = new Demo4();
		demo4.call();
		//这个里面没有Demo1、2的问题
		//因为这个
		try {
			Thread.sleep(5*1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		try {
			demo4.CountDownLatch.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		System.out.println("主线程内容");
	}

}
