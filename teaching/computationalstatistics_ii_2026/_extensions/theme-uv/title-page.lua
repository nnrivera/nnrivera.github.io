function Meta(meta)
  -- Quarto merges the selected format's title-layout option into metadata.
  local layout = meta['title-layout'] and pandoc.utils.stringify(meta['title-layout']) or 'lecture'
  if layout ~= 'lecture' and layout ~= 'talk' then
    error('title-layout: choose lecture or talk.')
  end
  meta['uv-talk-title'] = layout == 'talk'
  -- The meta shortcode cannot display a list. Keep a plain author list for
  -- the default footer, independently of the title page's affiliations.
  local names = {}
  if meta['by-author'] then
    for _, author in ipairs(meta['by-author']) do
      table.insert(names, pandoc.utils.stringify(author.name.literal))
    end
  elseif meta.author then
    if pandoc.utils.type(meta.author) == 'List' then
      for _, author in ipairs(meta.author) do
        table.insert(names, pandoc.utils.stringify(author))
      end
    else
      table.insert(names, pandoc.utils.stringify(meta.author))
    end
  end
  meta['uv-footer-author'] = pandoc.MetaString(table.concat(names, ', '))
  return meta
end
