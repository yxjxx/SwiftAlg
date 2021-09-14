//
//  Queue.swift
//  SwiftAlg
//
//  Created by yxj on 2021/9/12.
//

import Foundation

protocol QueueProtocol {
  associatedtype DataType: Comparable

  /**
  将一个新元素插入到队列中。
  - 参数 item：要添加的元素。
  - 返回值：插入是否成功。
  */
  @discardableResult func add(_ item: DataType) -> Bool

  /**
  删除首个元素。
  - 返回值：被移除的元素。
  - 抛出值：QueueError 类型的错误。
  */
  @discardableResult func remove() throws -> DataType

  /**
  获取到队列中的首个元素，并将其移出队列。
  - 返回值：一个包含队列中首个元素的可选值。
  */
  func dequeue() -> DataType?

  /**
  获取队列中的首个元素，但不将它移出队列。
  - 返回值：一个包含队列中首个元素的可选值。
  */
  func peek() -> DataType?

  /**
  清空队列。
  */
  func clear() -> Void
}

enum QueueError: Error {
  case noSuchItem(String)
}

/**
 基于堆数据结构的 PriorityQueue 实现。
*/

class PriorityQueue<DataType: Comparable> {

  /**
   队列的存储。
  */
  private var queue: Array<DataType>

  /**
   当前队列的大小。
  */
  public var size: Int {
    return self.queue.count
  }

  public init() {
    self.queue = Array<DataType>()
  }
}

private extension Int {
  var leftChild: Int {
    return (2 * self) + 1
  }

  var rightChild: Int {
    return (2 * self) + 2
  }

  var parent: Int {
    return (self - 1) / 2
  }
}

extension PriorityQueue: QueueProtocol {
  @discardableResult
  public func add(_ item: DataType) -> Bool {
    self.queue.append(item)
    self.heapifyUp(from: self.queue.count - 1)
    return true
  }

  @discardableResult
  public func remove() throws -> DataType {
    guard self.queue.count > 0 else {
      throw QueueError.noSuchItem("Attempt to remove item from an empty queue.")
    }
    return self.popAndHeapifyDown()
  }

  public func dequeue() -> DataType? {
    guard self.queue.count > 0 else {
      return nil
    }
    return self.popAndHeapifyDown()
  }

  public func peek() -> DataType? {
    return self.queue.first
  }

  public func clear() {
    self.queue.removeAll()
  }

  /**
  弹出队列中的第一个元素，并通过将根元素移向队尾的方式恢复最小堆排序。
  - 返回值: 队列中的第一个元素。
  */
  private func popAndHeapifyDown() -> DataType {
    let firstItem = self.queue[0]

    if self.queue.count == 1 {
      self.queue.remove(at: 0)
      return firstItem
    }

    self.queue[0] = self.queue.remove(at: self.queue.count - 1)

    self.heapifyDown()

    return firstItem
  }

  /**
   通过将元素移向队头的方式恢复最小堆排序。
   - 参数 index: 要移动的元素的索引值。
   */
  private func heapifyUp(from index: Int) {
    var child = index
    var parent = child.parent

    while parent >= 0 && self.queue[parent] > self.queue[child] {
      swap(parent, with: child)
      child = parent
      parent = child.parent
    }
  }

  /**
   通过将根元素移向队尾的方式恢复队列的最小堆排序。
   */
  private func heapifyDown() {
    var parent = 0

    while true {
      let leftChild = parent.leftChild
      if leftChild >= self.queue.count {
        break
      }

      let rightChild = parent.rightChild
      var minChild = leftChild
      if rightChild < self.queue.count && self.queue[minChild] > self.queue[rightChild] {
        minChild = rightChild
      }

      if self.queue[parent] > self.queue[minChild] {
        self.swap(parent, with: minChild)
        parent = minChild
      } else {
        break
      }
    }
  }

  /**
   交换存储中位于两处索引值位置的元素。
   - 参数 firstIndex：第一个要交换元素的索引。
   - 参数 secondIndex：第二个要交换元素的索引。
   */
  private func swap(_ firstIndex: Int, with secondIndex: Int) {
    queue.swapAt(firstIndex, secondIndex)
  }
}

extension PriorityQueue: CustomStringConvertible {
  public var description: String {
    return self.queue.description
  }
}
