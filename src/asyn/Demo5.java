package asyn;

import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class Demo5 extends BaseDemo {

	private CyclicBarrier cyclicBarrier = new CyclicBarrier(2);
	
	@Override
	public void callback(long response) {
		System.out.println("得到结果");
        System.out.println(response);
        System.out.println("调用结束");

        try {
            cyclicBarrier.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (BrokenBarrierException e) {
            e.printStackTrace();
        }
	}

	public static void main(String[] args) {
		Demo5 demo5 = new Demo5();

		demo5.call();
		
		try {
			Thread.sleep(5*1000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

        try {
            demo5.cyclicBarrier.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (BrokenBarrierException e) {
            e.printStackTrace();
        }

        System.out.println("主线程内容");
	}

}
