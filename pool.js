// pool.js — round-robin 작업 풀 (순수 로직, DOM·모델 없음).
// 발화를 작업자(인식 세션)들에게 번갈아 배정한다: 1번, 2번, 1번, 2번…
// 작업자가 바쁘면 그 차례는 기다리기만 하고 버려지지 않는다.
// (대화 모드 향상모드: 인식기를 2개로 복제해서 연속으로 말해도 안 버림.
//  Node 테스트: work/test_pool_v22.mjs)
export function createPool(workerCount) {
  if (workerCount < 1) workerCount = 1;
  const chains = Array.from({ length: workerCount }, () => Promise.resolve());
  let next = 0;
  return {
    workerCount,
    // task: (workerIndex) => Promise<결과>. 해당 작업자의 차례가 되면 실행한다.
    submit(task) {
      const idx = next;
      next = (next + 1) % workerCount;
      const run = chains[idx].then(() => task(idx));
      // 실패해도 체인이 끊기지 않게 (다음 차례는 계속 돈다).
      chains[idx] = run.catch(() => {});
      return run;
    },
  };
}
