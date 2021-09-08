//
//  MyHashTable.swift
//  SwiftAlg
//
//  Created by yxj on 2021/9/8.
//

import Foundation

//两个范型 Key 和 Value
public struct MyHashTable<Key: Hashable, Value> {

    private typealias Element = (key: Key, value: Value)
    private typealias Bukets = [Element]

    private var buckets: [Bukets] //存储位置
    var count = 0 //桶的容量

    var isEmpty: Bool { return count == 0 }

    init(_ capacity: Int) {
        buckets = Array<Bukets>.init(repeating: [], count: capacity)
    }

    //hasher 方法，除留余数法
    private func index(forKey key: Key) -> Int{
        return abs(key.hashValue) % buckets.count
    }

    func value(forKey key: Key) -> Value? {
        let index = index(forKey: key)
        for (i, element) in buckets[index].enumerated() {
            if element.key == key {
                let e = buckets[index][i].value
                return e
            }
        }
        return nil
    }

    public mutating func setValue(_ value: Value, forKey key: Key) {
        let index = index(forKey: key)
        for (i, element) in buckets[index].enumerated() {
            if element.key == key {
                buckets[index][i].value = value //hash 冲突解决链地址法
                return
            }
        }
        buckets[index].append((key, value))//不存在，直接插入
        count += 1
    }

    @discardableResult mutating func removeValue(forKey key: Key) -> Value? {
        let index = index(forKey: key)
        for (i, element) in buckets[index].enumerated() {
            if element.key == key {
                let e = buckets[index][i].value
                buckets[index].remove(at: i)
                count -= 1
                return e
            }
        }
        return nil
    }

    subscript(key: Key) -> Value? {
        get {
            value(forKey: key)
        }
        set {
            if let value = newValue {
                setValue(value, forKey: key)
            } else {
                removeValue(forKey: key)
            }
        }
    }
}
