// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/gcs/store_client/redis_store_client.h"

#include "ray/common/test_util.h"
#include "ray/gcs/redis_client.h"
#include "ray/gcs/store_client/test/store_client_test_base.h"

namespace ray {

namespace gcs {

class RedisStoreClientTest : public StoreClientTestBase {
 public:
  RedisStoreClientTest() {}

  virtual ~RedisStoreClientTest() {}

  static void SetUpTestCase() { TestSetupUtil::StartUpRedisServers(std::vector<int>()); }

  static void TearDownTestCase() { TestSetupUtil::ShutDownRedisServers(); }

  void InitStoreClient() override {
    RedisClientOptions options("127.0.0.1",
                               TEST_REDIS_SERVER_PORTS.front(),
                               "",
                               /*enable_sharding_conn=*/false);
    redis_client_ = std::make_shared<RedisClient>(options);
    RAY_CHECK_OK(redis_client_->Connect(io_service_pool_->GetAll()));

    store_client_ = std::make_shared<RedisStoreClient>(redis_client_);
  }

  void DisconnectStoreClient() override { redis_client_->Disconnect(); }

 protected:
  std::shared_ptr<RedisClient> redis_client_;
};

TEST_F(RedisStoreClientTest, AsyncPutAndAsyncGetTest) { TestAsyncPutAndAsyncGet(); }

TEST_F(RedisStoreClientTest, AsyncGetAllAndBatchDeleteTest) {
  TestAsyncGetAllAndBatchDelete();
}

TEST_F(RedisStoreClientTest, BasicSimple) {
  // Send 100 times write and then read
  auto cnt = std::make_shared<std::atomic<size_t>>(0);
  for (size_t i = 0; i < 100; ++i) {
    for (size_t j = 0; j < 20; ++j) {
      ++*cnt;
      ASSERT_TRUE(store_client_
                      ->AsyncPut("T",
                                 absl::StrCat("A", std::to_string(j)),
                                 std::to_string(i),
                                 true,
                                 [i, cnt](auto r) {
                                   --*cnt;
                                   ASSERT_TRUE((i == 0 && r) || (i != 0 && !r));
                                 })
                      .ok());
    }
  }
  for (size_t j = 0; j < 20; ++j) {
    ++*cnt;
    ASSERT_TRUE(store_client_
                    ->AsyncGet("T",
                               absl::StrCat("A", std::to_string(j)),
                               [cnt](auto s, auto r) {
                                 --*cnt;
                                 ASSERT_TRUE(r.has_value());
                                 ASSERT_EQ(*r, "99");
                               })
                    .ok());
  }
  ASSERT_TRUE(WaitForCondition([cnt]() { return *cnt == 0; }));
}

TEST_F(RedisStoreClientTest, Complicated) {
  int window = 10;
  std::atomic<size_t> finished{0};
  std::atomic<size_t> sent{0};

  for (int i = 0; i < 1000; i += window) {
    std::vector<std::string> keys;
    for (int j = i; j < i + window; ++j) {
      ++sent;
      RAY_LOG(INFO) << "S AsyncPut: " << ("P_" + std::to_string(j));
      ASSERT_TRUE(store_client_
                      ->AsyncPut("N",
                                 "P_" + std::to_string(j),
                                 std::to_string(j),
                                 true,
                                 [&finished, j](auto r) mutable {
                                   RAY_LOG(INFO)
                                       << "F AsyncPut: " << ("P_" + std::to_string(j));
                                   ++finished;
                                   ASSERT_TRUE(r);
                                 })
                      .ok());
      keys.push_back(std::to_string(j));
    }

    std::vector<std::string> p_keys;
    for (auto &key : keys) {
      p_keys.push_back("P_" + key);
    }

    std::vector<std::string> n_keys;
    for (auto &key : keys) {
      n_keys.push_back("N_" + key);
    }

    ++sent;
    RAY_LOG(INFO) << "S AsyncMultiGet: " << absl::StrJoin(p_keys, ",");
    ASSERT_TRUE(
        store_client_
            ->AsyncMultiGet(
                "N",
                p_keys,
                [&finished, i, keys, window, &sent, p_keys, n_keys, this](
                    auto m) mutable {
                  RAY_LOG(INFO) << "F SendAsyncMultiGet: " << absl::StrJoin(p_keys, ",");
                  ++finished;
                  ASSERT_EQ(keys.size(), m.size());
                  for (auto &key : keys) {
                    ASSERT_EQ(m["P_" + key], key);
                  }

                  if ((i / window) % 2 == 0) {
                    // Delete non exist keys
                    for (size_t i = 0; i < keys.size(); ++i) {
                      ++sent;
                      RAY_LOG(INFO) << "S AsyncDelete: " << n_keys[i];
                      ASSERT_TRUE(
                          store_client_
                              ->AsyncDelete("N",
                                            n_keys[i],
                                            [&finished, n_keys, i](auto b) mutable {
                                              RAY_LOG(INFO)
                                                  << "F AsyncDelete: " << n_keys[i];
                                              ++finished;
                                              ASSERT_FALSE(b);
                                            })
                              .ok());

                      ++sent;
                      RAY_LOG(INFO) << "S AsyncExists: " << p_keys[i];
                      ASSERT_TRUE(
                          store_client_
                              ->AsyncExists("N",
                                            p_keys[i],
                                            [&finished, p_keys, i](auto b) mutable {
                                              RAY_LOG(INFO)
                                                  << "F AsyncExists: " << p_keys[i];
                                              ++finished;
                                              ASSERT_TRUE(b);
                                            })
                              .ok());
                    }
                  } else {
                    ++sent;
                    RAY_LOG(INFO) << "S AsyncBatchDelete: " << absl::StrJoin(p_keys, ",");
                    ASSERT_TRUE(store_client_
                                    ->AsyncBatchDelete(
                                        "N",
                                        p_keys,
                                        [&finished, p_keys, keys](auto n) mutable {
                                          RAY_LOG(INFO) << "F AsyncBatchDelete: "
                                                        << absl::StrJoin(p_keys, ",");
                                          ++finished;
                                          ASSERT_EQ(n, keys.size());
                                        })
                                    .ok());

                    for (auto p_key : p_keys) {
                      ++sent;
                      RAY_LOG(INFO) << "S AsyncExists: " << p_key;
                      ASSERT_TRUE(store_client_
                                      ->AsyncExists("N",
                                                    p_key,
                                                    [&finished, p_key](auto b) mutable {
                                                      RAY_LOG(INFO)
                                                          << "F AsyncExists: " << p_key;
                                                      ++finished;
                                                      ASSERT_FALSE(false);
                                                    })
                                      .ok());
                    }
                  }
                })
            .ok());
  }
  ASSERT_TRUE(WaitForCondition(
      [&finished, &sent]() {
        RAY_LOG(INFO) << finished << "/" << sent;
        return finished == sent;
      },
      5000));
}

TEST_F(RedisStoreClientTest, Random) {
  std::map<std::string, std::string> dict;
  std::atomic<size_t> counter = 0;
  auto m_gen_keys = []() {
    auto num_keys = std::rand() % 10;
    std::unordered_set<std::string> keys;
    while (keys.size() < num_keys) {
      auto k = std::to_string(std::rand() % 1000);
      keys.insert(k);
    }
    return std::vector<std::string>(keys.begin(), keys.end());
  };

  auto m_get = [&counter, this]() mutable {
    auto keys = m_gen_keys();
    absl::flat_hash_map<std::string, std::string> result;
    for (auto key : keys) {
      if (dict.count(key)) {
        result[key] = dict[key];
      }
    }
    counter += 1;
    RAY_CHECK_OK(
        store_client_->AsyncMultiGet("N", keys, [result, &counter](auto m) mutable {
          counter -= 1;
          ASSERT_TRUE(m == result);
        }));
  };

  auto m_batch_delete = [&counter, this]() mutable {
    auto keys = m_gen_keys();
    size_t deleted_num = 0;
    for (auto key : keys) {
      deleted_num += dict.erase(key);
    }
    counter += 1;
    RAY_CHECK_OK(store_client_->AsyncBatchDelete(
        "N", keys, [&counter, deleted_num](auto v) mutable {
          counter -= 1;
          ASSERT_EQ(v, deleted_num)
        }));
  };

  auto m_delete = [&counter, this]() mutable {
    auto k = std::to_string(std::rand() % 1000);
    bool deleted = dict.erase(k) > 0;
    counter += 1;
    RAY_CHECK_OK(store_client_->AsyncBatchDelete("N", k, [&counter, deleted](auto r) {
      counter -= 1;
      ASSERT_EQ(deleted, r);
    }));
  };

  auto m_get = [&counter, this]() mutable {
    auto k = std::to_string(std::rand() % 1000);
    boost::optional<std::string> v;
    if (dict.count(k)) {
      r = dict[k];
    }
    counter += 1;
    RAY_CHECK_OK(store_client_->AsyncGet("N", k, [&counter, v](auto r) {
      counter -= 1;
      ASSERT_EQ(v, r);
    }));
  };

  auto m_exists = [&counter, this]() mutable {
    auto k = std::to_string(std::rand() % 1000);
    bool existed = dict.count(k);
    counter += 1;
    RAY_CHECK_OK(store_client_->AsyncExists("N", k, [v, &counter](auto r) mutable {
      counter -= 1;
      ASSERT_EQ(v, r);
    }));
  };

  std::vector<std::function<void()>> ops{m_batch_delete, m_delete, m_get, m_exists};

  for (size_t i = 0; i < 1000; ++i) {
    auto idx = std::rand() % ops.size();
    ops[idx]();
  }
}

}  // namespace gcs

}  // namespace ray

int main(int argc, char **argv) {
  InitShutdownRAII ray_log_shutdown_raii(ray::RayLog::StartRayLog,
                                         ray::RayLog::ShutDownRayLog,
                                         argv[0],
                                         ray::RayLogLevel::INFO,
                                         /*log_dir=*/"");
  ::testing::InitGoogleTest(&argc, argv);
  RAY_CHECK(argc == 3);
  ray::TEST_REDIS_SERVER_EXEC_PATH = argv[1];
  ray::TEST_REDIS_CLIENT_EXEC_PATH = argv[2];
  return RUN_ALL_TESTS();
}
