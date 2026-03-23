function Header(el)
  if el.level > 1 then
    el.level = el.level - 1
  end
  return el
end
