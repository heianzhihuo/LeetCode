package asyn;

import java.util.concurrent.locks.*;

public class Demo2 extends BaseDemo {
	private final Lock lock = new ReentrantLock();
	private final Condition con = lock.newCondition();
	private int cons = 0;
	
	@Override
	public void callback(long response) {
		System.out.println("得到结果");
		System.out.println(response);
		System.out.println("调用结束");
		lock.lock();
		try {
			cons++;
			con.signal();
		} finally {
			lock.unlock();
		}
	}

	public static void main(String[] args) {
		Demo2 demo2 = new Demo2();
		//这里加锁存在一个问题,如果子线程比主线程先执行完成
		//即callback先执行完成，此时已经调用了con.signal()
		//而主线程后执行con.await()，此时将发生无限期等待
		//因此改进后，添加了一个条件，这种无限期等待
		demo2.call();
		try {
			Thread.sleep(5*1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		demo2.lock.lock();
		try {
			if(demo2.cons==0)
				demo2.con.await();
		} catch (InterruptedException e) {
			e.printStackTrace();
		} finally {
			demo2.lock.unlock();
		}
		System.out.println("主线程内容");
	}

}
