# Test Patterns

## Fixture Pattern

```python
@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_user():
    return User(name="Alice", role="admin")
```

## Parametrize Pattern

```python
@pytest.mark.parametrize("input_val, expected", [
    (0, True),
    (1, True),
    (-1, False),
])
def test_is_non_negative(input_val, expected):
    assert is_non_negative(input_val) == expected
```

## Async Test Pattern

```python
async def test_fetch_user():
    store = InMemoryStore()
    user = await store.get_user("alice")
    assert user.name == "Alice"
```
